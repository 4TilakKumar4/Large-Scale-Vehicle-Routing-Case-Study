"""
VRP_ORTools_Overnight_C.py — OR-Tools overnight routing with CP-SAT day
assignment (Q3 Option C).

Extends Option B by replacing the greedy angular sweep with a formal CP-SAT
integer program for day assignment.  The routing layer (ORToolsOvernightSolver)
is identical across Options A, B, and C — only the day-assignment method
changes.

Pipeline:
  1. CP-SAT (cpSatAssigner.py)   — integer-program day assignment
  2. OR-Tools RoutingModel       — two-day overnight routing per adjacent pair
  3. 10-hour break intervals      — DOT HOS compliance

Requires VRP_DataAnalysis.py to have been run first (data/ must exist).

Usage:
    python VRP_ORTools_Overnight_C.py                 # solver + map
    python VRP_ORTools_Overnight_C.py --no-map        # solver only
    python VRP_ORTools_Overnight_C.py --time 60       # routing time limit/pair
    python VRP_ORTools_Overnight_C.py --cp-time 60    # CP-SAT time limit
    python VRP_ORTools_Overnight_C.py --compare       # compare B vs C

Outputs:
    outputs/or_tools_overnight_c/route_details.csv
    outputs/or_tools_overnight_c/resource_summary.csv
    outputs/or_tools_overnight_c/driver_chains.csv
    outputs/or_tools_overnight_c/assignment_summary.csv  ← CP-SAT diagnostics
    outputs/or_tools_overnight_c/routes_map_ortools_c.html
"""

import argparse
import os

import pandas as pd

from vrp_solvers.base import (
    DATA_DIR,
    DAYS,
    DEPOT_ZIP,
    detailedRouteTrace,
    evaluateRoute,
    getDistance,
    loadInputs,
    routeIds,
)
from vrp_solvers.cpSatAssigner          import CpSatAssigner
from vrp_solvers.angularSweepAssigner   import AngularSweepAssigner, loadAngularSweepInputs
from vrp_solvers.orToolsOvernightSolver import ORToolsOvernightSolver
from vrp_solvers.orToolsSolver          import ORToolsSolver
from vrp_solvers.overnightSolver        import ADJACENT_DAY_PAIRS
from vrp_solvers.resourceAnalyser       import ResourceAnalyser
from vrp_solvers.costModel              import CostModel

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "or_tools_overnight_c")
MAP_FILE   = os.path.join(OUTPUT_DIR, "routes_map_ortools_c.html")
COUNTRY_CODE = "US"

ROUTE_PALETTE = [
    "#E63946", "#F4A261", "#2A9D8F", "#457B9D",
    "#9B5DE5", "#F72585", "#4CC9F0", "#06D6A0",
]

DAY_COLORS = {
    "Mon": "#E63946",
    "Tue": "#F4A261",
    "Wed": "#2A9D8F",
    "Thu": "#457B9D",
    "Fri": "#9B5DE5",
}


def printAssignmentSummary(summaryDF, originalOrders, revisedOrders, solveStatus):
    """Print CP-SAT assignment diagnostics."""
    print("\nCP-SAT Day Assignment")
    print("-" * 60)
    print(f"  Solver status: {solveStatus}")

    origCube = originalOrders.groupby("DayOfWeek")["CUBE"].sum().reindex(DAYS)
    newCube  = revisedOrders.groupby("DayOfWeek")["CUBE"].sum().reindex(DAYS)
    target   = originalOrders["CUBE"].sum() / len(DAYS)

    print(f"\n  {'Day':<6} {'Original':>10} {'CP-SAT':>10} {'Target':>10} {'Balance%':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for day in DAYS:
        before = origCube.get(day, 0)
        after  = newCube.get(day, 0)
        pct    = after / target * 100 if target else 0
        print(f"  {day:<6} {before:>10,.0f} {after:>10,.0f} {target:>10,.0f} {pct:>9.1f}%")

    merged = originalOrders[["ORDERID", "DayOfWeek"]].merge(
        revisedOrders[["ORDERID", "DayOfWeek"]], on="ORDERID", suffixes=("_orig", "_new")
    )
    moved = (merged["DayOfWeek_orig"] != merged["DayOfWeek_new"]).sum()
    print(f"\n  Orders reassigned: {moved} / {len(originalOrders)}")

    # Affinity: how many orders stayed on their angularly-preferred day
    # (This requires rerunning the sweep — use summaryDF pct_of_target as proxy)
    balance = summaryDF["pct_of_target"]
    print(f"  Cube balance range: {balance.min():.1f}% – {balance.max():.1f}% of target")


def printDayReport(day, routes, label="OR-Tools C"):
    print(f"\n{day}  [{label}]")
    print("-" * 60)
    dayMiles = 0
    for i, route in enumerate(routes, start=1):
        r = evaluateRoute(route)
        print(
            f"  Route {i}: {routeIds(route)} | "
            f"orders={len(route)} | miles={r['total_miles']} | "
            f"cube={r['total_cube']} | drive={r['total_drive']}h | "
            f"duty={r['total_duty']}h | feasible={r['overall_feasible']}"
        )
        dayMiles += r["total_miles"]
    print(f"  {day} routes: {len(routes)} | total miles: {dayMiles}")
    return dayMiles


def printCompareTable(statsB, statsC, labelB="Option B", labelC="Option C"):
    """Option B vs Option C side-by-side per pair."""
    print(f"\n{labelB} vs {labelC} Comparison")
    print("-" * 60)
    print(f"  {'Pair':<10} {labelB:>12} {labelC:>12} {'Saved':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")

    totalB = totalC = 0
    for pairKey in statsC:
        milesB = statsB.get(pairKey, {}).get("miles", 0)
        milesC = statsC[pairKey]["miles"]
        saved  = milesB - milesC
        pct    = (saved / milesB * 100) if milesB else 0
        totalB += milesB
        totalC += milesC
        print(f"  {pairKey:<10} {milesB:>12,} {milesC:>12,} {saved:>+9,} ({pct:+.1f}%)")

    saved = totalB - totalC
    pct   = (saved / totalB * 100) if totalB else 0
    print(f"  {'TOTAL':<10} {totalB:>12,} {totalC:>12,} {saved:>+9,} ({pct:+.1f}%)")
    print(f"\n  Positive saved = Option C used fewer miles than B")


def exportRouteDetails(routesByDay):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    locsPath = os.path.join(DATA_DIR, "locations_clean.csv")
    locs     = pd.read_csv(locsPath) if os.path.exists(locsPath) else None

    allRows = []
    for day in DAYS:
        for routeNum, route in enumerate(routesByDay.get(day, []), start=1):
            allRows.extend(detailedRouteTrace(route, day, routeNum, locs))

    detailDF = pd.DataFrame(allRows, columns=[
        "day", "route_number", "stop_sequence", "order_id",
        "location", "arrival_time", "departure_time", "delivery_volume_cuft",
    ])
    detailDF.to_csv(os.path.join(OUTPUT_DIR, "route_details.csv"), index=False)
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'route_details.csv')}")


def exportResourceReport(analyser):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summaryDF, chainsDF = analyser.toDataFrame()
    summaryDF.to_csv(os.path.join(OUTPUT_DIR, "resource_summary.csv"), index=False)
    chainsDF.to_csv(os.path.join(OUTPUT_DIR,  "driver_chains.csv"),    index=False)
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'resource_summary.csv')}")
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'driver_chains.csv')}")


def exportAssignmentSummary(summaryDF):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "assignment_summary.csv")
    summaryDF.to_csv(path, index=False)
    print(f"  Saved: {path}")


def geocodeAllZips(allZips):
    import pgeocode
    nomi   = pgeocode.Nominatim(COUNTRY_CODE)
    coords = {}
    for z in allZips:
        zipStr = str(int(z)).zfill(5)
        result = nomi.query_postal_code(zipStr)
        if not pd.isna(result.latitude) and not pd.isna(result.longitude):
            coords[z] = (float(result.latitude), float(result.longitude))

    missing = [z for z in allZips if z not in coords]
    if missing:
        coords.update(_mdsLayout(list(allZips), missing))
    print(f"  Geocoded {len(coords)}/{len(allZips)} ZIPs")
    return coords


def _mdsLayout(allZips, missing):
    import numpy as np
    from sklearn.manifold import MDS

    n = len(allZips)
    D = np.zeros((n, n))
    for i, za in enumerate(allZips):
        for j, zb in enumerate(allZips):
            if i != j:
                try:
                    D[i, j] = float(getDistance(za, zb))
                except Exception:
                    D[i, j] = 999

    pos  = MDS(n_components=2, dissimilarity="precomputed",
               random_state=42, normalized_stress="auto").fit_transform(D)
    span = pos.max(axis=0) - pos.min(axis=0)
    span[span == 0] = 1
    result = {}
    for i, z in enumerate(allZips):
        if z in missing:
            norm      = (pos[i] - pos.min(axis=0)) / span
            result[z] = (42.3 + (norm[1] - 0.5) * 1.5,
                         -71.1 + (norm[0] - 0.5) * 2.0)
    return result


def buildMap(routesByDay, zipCoords):
    import folium
    from folium import plugins

    depotCoord = zipCoords.get(DEPOT_ZIP, (42.3, -71.1))
    m = folium.Map(location=depotCoord, zoom_start=9, tiles="CartoDB positron")

    folium.Marker(
        location=depotCoord,
        tooltip=f"<b>DEPOT</b><br>ZIP {DEPOT_ZIP}",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)

    for day in DAYS:
        dayGroup = folium.FeatureGroup(name=f"{day}", show=True)
        for routeIdx, route in enumerate(routesByDay.get(day, [])):
            routeColor = ROUTE_PALETTE[routeIdx % len(ROUTE_PALETTE)]
            r          = evaluateRoute(route)
            routeLabel = f"{day} · Route {routeIdx + 1}"

            waypoints = [depotCoord]
            for stop in route:
                z = stop["TOZIP"]
                if z in zipCoords:
                    waypoints.append(zipCoords[z])
            waypoints.append(depotCoord)

            folium.PolyLine(
                locations=waypoints, color=routeColor, weight=3, opacity=0.85,
                tooltip=(f"{routeLabel} | {r['total_miles']} mi | "
                         f"{len(route)} orders | feasible={r['overall_feasible']}"),
            ).add_to(dayGroup)

            plugins.AntPath(
                locations=waypoints, color=routeColor, weight=3,
                opacity=0.5, delay=1200, dash_array=[10, 40],
            ).add_to(dayGroup)

            for seq, stop in enumerate(route, start=1):
                z = stop["TOZIP"]
                if z not in zipCoords:
                    continue
                coord     = zipCoords[z]
                oid       = int(stop["ORDERID"])
                popupHtml = (
                    f'<div style="font-family:monospace;font-size:13px;min-width:180px;">'
                    f"<b>{routeLabel}</b><br>Stop #{seq}<br>"
                    f"Order ID: {oid}<br>ZIP: {int(z)}<br>"
                    f"Cube: {int(stop['CUBE'])} ft³</div>"
                )
                folium.CircleMarker(
                    location=coord, radius=7, color="white", weight=2,
                    fill=True, fill_color=routeColor, fill_opacity=0.9,
                    tooltip=f"Stop {seq} · Order {oid}",
                    popup=folium.Popup(popupHtml, max_width=230),
                ).add_to(dayGroup)

        dayGroup.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    legendHtml = (
        '<div style="position:fixed;bottom:30px;left:30px;z-index:1000;'
        'background:white;border-radius:8px;padding:12px 16px;'
        'box-shadow:0 2px 10px rgba(0,0,0,0.15);'
        'font-family:Segoe UI,sans-serif;font-size:13px;">'
        '<b style="font-size:14px;">Days (CP-SAT C)</b><br><br>'
    )
    for day, color in DAY_COLORS.items():
        legendHtml += (
            f'<div style="display:flex;align-items:center;margin-bottom:5px;">'
            f'<div style="width:14px;height:14px;border-radius:50%;background:{color};'
            f'margin-right:8px;flex-shrink:0;"></div>{day}</div>'
        )
    legendHtml += "</div>"
    m.get_root().html.add_child(folium.Element(legendHtml))
    return m


def _runOptionB(originalOrders, locs, routingTimeSec):
    """Run Option B (sweep + overnight) for comparison stats."""
    sweeper      = AngularSweepAssigner(originalOrders, locs)
    revisedB     = sweeper.assign()
    solverB      = ORToolsOvernightSolver(timeLimitSec=routingTimeSec)

    for d1, d2 in ADJACENT_DAY_PAIRS:
        solverB.solvePair(
            d1, revisedB[revisedB["DayOfWeek"] == d1].copy(),
            d2, revisedB[revisedB["DayOfWeek"] == d2].copy(),
        )
    return solverB.getStats()


def main():
    parser = argparse.ArgumentParser(
        description="NHG OR-Tools Overnight — Q3 Option C (CP-SAT assignment + overnight)"
    )
    parser.add_argument("--time",    type=int, default=60,
                        help="Per-pair routing time limit in seconds (default: 60)")
    parser.add_argument("--cp-time", type=int, default=30,
                        help="CP-SAT assignment time limit in seconds (default: 30)")
    parser.add_argument("--compare", action="store_true",
                        help="Also run Option B and print B vs C comparison")
    parser.add_argument("--no-map",  action="store_true",
                        help="Skip geocoding and map generation")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: load data ───────────────────────────────────────────────
    originalOrders, _ = loadInputs()
    _, locs           = loadAngularSweepInputs()

    # ── Step 2: CP-SAT day assignment ───────────────────────────────────
    print(f"Running CP-SAT day assignment (time limit: {args.cp_time}s)...")
    assigner      = CpSatAssigner(originalOrders, locs, timeLimitSec=args.cp_time)
    revisedOrders = assigner.assign()
    summaryDF     = assigner.getSectorSummary()
    solveStatus   = assigner.getSolveStatus()

    printAssignmentSummary(summaryDF, originalOrders, revisedOrders, solveStatus)

    # ── Step 3: solve adjacent pairs with CP-SAT assignments ───────────
    solver      = ORToolsOvernightSolver(timeLimitSec=args.time)
    routesByDay = {day: [] for day in DAYS}

    for d1, d2 in ADJACENT_DAY_PAIRS:
        d1Orders = revisedOrders[revisedOrders["DayOfWeek"] == d1].copy()
        d2Orders = revisedOrders[revisedOrders["DayOfWeek"] == d2].copy()

        print(f"Solving pair {d1}-{d2}  ({len(d1Orders)} + {len(d2Orders)} orders, "
              f"limit={args.time}s)...")
        pairRoutes = solver.solvePair(d1, d1Orders, d2, d2Orders)

        if pairRoutes[d1]:
            routesByDay[d1] = pairRoutes[d1]
        if pairRoutes[d2]:
            routesByDay[d2] = pairRoutes[d2]

    # Friday fallback
    if not routesByDay["Fri"]:
        print("Solving Fri standalone...")
        friSolver = ORToolsSolver(timeLimitSec=args.time)
        routesByDay["Fri"] = friSolver.solve(
            revisedOrders[revisedOrders["DayOfWeek"] == "Fri"].copy()
        )

    # ── Step 4: report ──────────────────────────────────────────────────
    print("\nOR-Tools Overnight C Routes")
    print("-" * 60)
    weeklyMiles  = 0
    weeklyRoutes = 0
    weeklyOrders = 0
    for day in DAYS:
        dayMiles  = printDayReport(day, routesByDay[day])
        weeklyMiles  += dayMiles
        weeklyRoutes += len(routesByDay[day])
        weeklyOrders += sum(len(r) for r in routesByDay[day])

    print("\nWeekly Summary")
    print("-" * 60)
    print(f"  Total routes:           {weeklyRoutes}")
    print(f"  Total orders fulfilled: {weeklyOrders}")
    print(f"  Total miles:            {weeklyMiles:,}")
    print(f"  Annual miles:           {weeklyMiles * 52:,}")

    if args.compare:
        print("\nRunning Option B for comparison...")
        statsB = _runOptionB(originalOrders, locs, args.time)
        printCompareTable(statsB, solver.getStats())

    analyser = ResourceAnalyser(routesByDay)
    analyser.analyse()
    analyser.printReport()

    costModel = CostModel()
    breakdown = costModel.weeklyBreakdown(routesByDay)
    costModel.printSummary(breakdown, label="OR-Tools Overnight C (CP-SAT + Overnight)")

    # ── Step 5: export ──────────────────────────────────────────────────
    print("\nExporting outputs...")
    exportRouteDetails(routesByDay)
    exportResourceReport(analyser)
    exportAssignmentSummary(summaryDF)

    if not args.no_map:
        print("\nGeocoding ZIPs...")
        allZips = {DEPOT_ZIP}
        for routes in routesByDay.values():
            for route in routes:
                for stop in route:
                    allZips.add(stop["TOZIP"])
        zipCoords = geocodeAllZips(allZips)
        print("Building map...")
        buildMap(routesByDay, zipCoords).save(MAP_FILE)
        print(f"  Saved: {MAP_FILE}")
    else:
        print("\n(Map skipped — run without --no-map to generate)")


if __name__ == "__main__":
    main()
