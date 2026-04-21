"""
VRP_RelaxedSchedule.py — Sub-problem 3: Relaxed delivery-day scheduling.

Two approaches compared:
  1. Angular sweep seed + greedy LS        (8–15% improvement expected)
  2. ALNS inter-day + greedy LS            (12–20% improvement expected)

Requires VRP_DataAnalysis.py to have been run first (data/ must exist).

Usage:
    python VRP_RelaxedSchedule.py              # both methods + map
    python VRP_RelaxedSchedule.py --no-map    # solver only (faster)

Outputs:
    outputs/relaxed_schedule/results_summary.csv    — weekly miles and cost per method
    outputs/relaxed_schedule/route_details_*.csv    — per-stop timing per method
    outputs/relaxed_schedule/resource_summary_*.csv — resource metrics per method
    outputs/relaxed_schedule/driver_chains_*.csv    — driver chains per method
    outputs/relaxed_schedule/day_assignments_*.csv  — final day assignment per method
    outputs/relaxed_schedule/routes_map_*.html      — Folium maps (unless --no-map)
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
    loadZipCoords,
    routeIds,
)
from vrp_solvers.relaxedScheduleSolver import (
    SweepRelaxedSolver,
    ALNSRelaxedSolver,
    getVisitGroups,
)
from vrp_solvers.resourceAnalyser import ResourceAnalyser
from vrp_solvers.costModel        import CostModel

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "relaxed_schedule")
COUNTRY_CODE = "US"

DAY_COLORS = {
    "Mon": "#E63946", "Tue": "#F4A261", "Wed": "#2A9D8F",
    "Thu": "#457B9D", "Fri": "#9B5DE5",
}
ROUTE_PALETTE = [
    "#E63946", "#F4A261", "#2A9D8F", "#457B9D",
    "#9B5DE5", "#F72585", "#4CC9F0", "#06D6A0",
]


def printScheduleSummary(title, routesByDay):
    print(f"\n{title}")
    print("-" * 60)
    totalMiles = totalRoutes = 0
    for day in DAYS:
        dayMiles = sum(evaluateRoute(r)["total_miles"] for r in routesByDay[day])
        n        = len(routesByDay[day])
        print(f"  {day}: routes={n} | miles={dayMiles:,}")
        totalMiles  += dayMiles
        totalRoutes += n
    print("-" * 60)
    print(f"  Weekly routes: {totalRoutes} | Weekly miles: {totalMiles:,}")
    return totalMiles, totalRoutes


def printMoves(label, moves):
    print(f"\nAccepted day reassignments — {label}")
    print("-" * 60)
    if not moves:
        print("  No improving moves found.")
        return
    for i, m in enumerate(moves, start=1):
        print(f"  {i}. Store {m['store']} | {m['from_day']} → {m['to_day']} | "
              f"orders {m['order_ids']}")


def exportRouteDetails(routesByDay, label):
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
    path = os.path.join(OUTPUT_DIR, f"route_details_{label}.csv")
    detailDF.to_csv(path, index=False)
    print(f"  Saved: {path}")


def exportResourceReport(analyser, label):
    summaryDF, chainsDF = analyser.toDataFrame()
    sp = os.path.join(OUTPUT_DIR, f"resource_summary_{label}.csv")
    cp = os.path.join(OUTPUT_DIR, f"driver_chains_{label}.csv")
    summaryDF.to_csv(sp, index=False)
    chainsDF.to_csv(cp,  index=False)
    print(f"  Saved: {sp}")
    print(f"  Saved: {cp}")


def exportDayAssignments(finalOrders, label):
    df = (
        finalOrders[["ORDERID", "TOZIP", "DayOfWeek"]]
        .sort_values(["DayOfWeek", "TOZIP", "ORDERID"])
        .reset_index(drop=True)
    )
    path = os.path.join(OUTPUT_DIR, f"day_assignments_{label}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def geocodeAllZips(allZips):
    import numpy as np
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
        mds = _mdsLayout(list(allZips))
        for z in missing:
            coords[z] = mds[z]
    print(f"  Geocoded {len(coords)}/{len(allZips)} ZIPs")
    return coords


def _mdsLayout(allZips):
    import numpy as np
    from sklearn.manifold import MDS
    n = len(allZips)
    D = np.zeros((n, n))
    for i, za in enumerate(allZips):
        for j, zb in enumerate(allZips):
            if i != j:
                try:   D[i, j] = float(getDistance(za, zb))
                except: D[i, j] = 999
    pos  = MDS(n_components=2, dissimilarity="precomputed",
               random_state=42, normalized_stress="auto").fit_transform(D)
    span = pos.max(axis=0) - pos.min(axis=0)
    span[span == 0] = 1
    coords = {}
    for i, z in enumerate(allZips):
        norm = (pos[i] - pos.min(axis=0)) / span
        coords[z] = (42.3 + (norm[1] - 0.5) * 1.5,
                     -71.1 + (norm[0] - 0.5) * 2.0)
    return coords


def buildMap(routesByDay, zipCoords, title):
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
        for rIdx, route in enumerate(routesByDay[day]):
            color     = ROUTE_PALETTE[rIdx % len(ROUTE_PALETTE)]
            r         = evaluateRoute(route)
            label     = f"{day} · Route {rIdx + 1}"
            waypoints = ([depotCoord]
                         + [zipCoords[s["TOZIP"]] for s in route if s["TOZIP"] in zipCoords]
                         + [depotCoord])
            folium.PolyLine(
                locations=waypoints, color=color, weight=3, opacity=0.85,
                tooltip=(f"{label} | {r['total_miles']} mi | "
                         f"{len(route)} orders | duty={r['total_duty']}h"),
            ).add_to(dayGroup)
            plugins.AntPath(locations=waypoints, color=color, weight=3,
                            opacity=0.5, delay=1200, dash_array=[10, 40]).add_to(dayGroup)
            for seq, stop in enumerate(route, start=1):
                z = stop["TOZIP"]
                if z not in zipCoords: continue
                coord     = zipCoords[z]
                oid       = int(stop["ORDERID"])
                popupHtml = (f'<div style="font-family:monospace;font-size:13px;">'
                             f"<b>{label}</b><br>Stop #{seq}<br>Order {oid}<br>"
                             f"ZIP {int(z)}<br>Cube {int(stop['CUBE'])} ft³</div>")
                folium.CircleMarker(
                    location=coord, radius=7, color="white", weight=2,
                    fill=True, fill_color=color, fill_opacity=0.9,
                    tooltip=f"Stop {seq} · Order {oid}",
                    popup=folium.Popup(popupHtml, max_width=220),
                ).add_to(dayGroup)
                folium.Marker(location=coord, icon=folium.DivIcon(
                    html=(f'<div style="font-size:9px;font-weight:bold;color:white;'
                          f'text-align:center;line-height:14px;">{seq}</div>'),
                    icon_size=(14, 14), icon_anchor=(7, 7))).add_to(dayGroup)
        dayGroup.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;top:10px;left:50px;z-index:1000;'
        f'background:white;padding:8px 12px;border-radius:6px;'
        f'box-shadow:0 2px 8px rgba(0,0,0,0.15);font-family:sans-serif;font-size:13px;">'
        f'<b>{title}</b></div>'
    ))
    return m


def runMethod(solver, orders, label, verbose=True):
    """Run one solver and return its outputs."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    routesByDay = solver.solve(orders)
    stats       = solver.getStats()
    finalOrders = solver.getOrders()
    moves       = solver.getMoves()

    weeklyMiles, weeklyRoutes = printScheduleSummary(f"Result — {label}", routesByDay)
    printMoves(label, moves)

    analyser = ResourceAnalyser(routesByDay)
    analyser.analyse()
    analyser.printReport()

    cm       = CostModel()
    bd       = cm.weeklyBreakdown(routesByDay)
    cm.printSummary(bd, label=label)

    return routesByDay, finalOrders, stats, analyser, bd


def main():
    parser = argparse.ArgumentParser(description="NHG Sub-problem 3 — Relaxed Schedule")
    parser.add_argument("--no-map", action="store_true",
                        help="Skip geocoding and map generation")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    orders, _ = loadInputs()
    loadZipCoords()

    from vrp_solvers.base import solveOneDay as _solve
    baselineRoutes = {day: _solve(orders[orders["DayOfWeek"] == day].copy()) for day in DAYS}
    baselineMiles  = sum(evaluateRoute(r)["total_miles"]
                        for d in DAYS for r in baselineRoutes[d])
    print(f"\nBaseline (historical fixed schedule): {baselineMiles:,} miles/week")

    METHODS = [
        ("sweep_ls",    SweepRelaxedSolver(verbose=True)),
        ("alns_ls",     ALNSRelaxedSolver(verbose=True)),
    ]

    summaryRows = []
    allZips     = {DEPOT_ZIP}

    for label, solver in METHODS:
        routesByDay, finalOrders, stats, analyser, bd = runMethod(solver, orders, label)

        report = analyser.getReport()
        summaryRows.append({
            "method":            label,
            "weekly_miles":      stats["weekly_miles"],
            "annual_miles":      stats["annual_miles"],
            "pct_vs_baseline":   round(
                (stats["weekly_miles"] - baselineMiles) / baselineMiles * 100, 2
            ),
            "routes":            stats["routes"],
            "moves_accepted":    stats["moves_accepted"],
            "min_drivers":       report["min_drivers"],
            "peak_trucks":       report["min_trucks_peak"],
            "avg_duty_hrs":      report["avg_weekly_duty_hrs"],
            "weekly_cost":       bd["weekly"]["total"],
            "annual_cost":       bd["annual"]["total"],
            "runtime_s":         stats["runtime_s"],
        })

        print(f"\nExporting {label}...")
        exportRouteDetails(routesByDay, label)
        exportResourceReport(analyser, label)
        exportDayAssignments(finalOrders, label)

        for d in DAYS:
            for route in routesByDay[d]:
                for stop in route:
                    allZips.add(stop["TOZIP"])

    summaryDF = pd.DataFrame(summaryRows)
    sp        = os.path.join(OUTPUT_DIR, "results_summary.csv")
    summaryDF.to_csv(sp, index=False)
    print(f"\n  Saved: {sp}")

    print("\nResults Summary")
    print("-" * 70)
    for _, row in summaryDF.iterrows():
        sign = "+" if row["pct_vs_baseline"] >= 0 else ""
        print(f"  {row['method']:<14} | miles={row['weekly_miles']:>7,} | "
              f"vs baseline={sign}{row['pct_vs_baseline']:.1f}% | "
              f"annual=${row['annual_cost']:>10,.0f} | "
              f"{row['runtime_s']:.0f}s")

    if not args.no_map:
        print("\nGeocoding ZIPs...")
        zipCoords = geocodeAllZips(allZips)
        print("Building maps...")
        for label, solver in METHODS:
            routesByDay = solver.solve(orders)   # re-use cached result implicitly via orders
            # rebuild from final assignments
            finalOrders = solver.getOrders()
            rb = {day: _solve(finalOrders[finalOrders["DayOfWeek"] == day].copy())
                  for day in DAYS}
            mapPath = os.path.join(OUTPUT_DIR, f"routes_map_{label}.html")
            buildMap(rb, zipCoords, label).save(mapPath)
            print(f"  Saved: {mapPath}")
    else:
        print("\n(Maps skipped — run without --no-map to generate)")


if __name__ == "__main__":
    main()