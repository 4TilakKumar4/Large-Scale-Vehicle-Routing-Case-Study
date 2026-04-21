"""
VRP_ORTools.py — Google OR-Tools CVRP on the NHG dataset (Q1: base case).

Requires VRP_DataAnalysis.py to have been run first (data/ must exist).

Usage:
    python VRP_ORTools.py                  # solver only
    python VRP_ORTools.py --time 60        # set per-day time limit in seconds
    python VRP_ORTools.py --compare        # run CW alongside OR-Tools and print diff
    python VRP_ORTools.py --no-map         # skip map generation

Outputs:
    outputs/or_tools/route_details.csv      — per-stop timing in Table 3 format
    outputs/or_tools/resource_summary.csv   — headline resource metrics
    outputs/or_tools/driver_chains.csv      — driver chain assignments
    outputs/or_tools/routes_map_ortools.html — interactive Folium map (unless --no-map)
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
from vrp_solvers.orToolsSolver   import ORToolsSolver
from vrp_solvers.clarkeWright    import ClarkeWrightSolver
from vrp_solvers.resourceAnalyser import ResourceAnalyser
from vrp_solvers.costModel        import CostModel

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "or_tools")
MAP_FILE   = os.path.join(OUTPUT_DIR, "routes_map_ortools.html")
COUNTRY_CODE = "US"

DAY_COLORS = {
    "Mon": "#E63946",
    "Tue": "#F4A261",
    "Wed": "#2A9D8F",
    "Thu": "#457B9D",
    "Fri": "#9B5DE5",
}

ROUTE_PALETTE = [
    "#E63946", "#F4A261", "#2A9D8F", "#457B9D",
    "#9B5DE5", "#F72585", "#4CC9F0", "#06D6A0",
]


def printDayReport(day, routes, label="OR-Tools"):
    """Print per-route details for one day; return (total_miles, total_orders)."""
    print(f"\n{day}  [{label}]")
    print("-" * 60)
    dayMiles  = 0
    dayOrders = 0

    for i, route in enumerate(routes, start=1):
        r = evaluateRoute(route)
        print(
            f"  Route {i}: {routeIds(route)} | "
            f"orders={len(route)} | miles={r['total_miles']} | "
            f"cube={r['total_cube']} | drive={r['total_drive']}h | "
            f"duty={r['total_duty']}h | "
            f"cap={r['capacity_feasible']} dot={r['dot_feasible']} "
            f"win={r['window_feasible']} | feasible={r['overall_feasible']}"
        )
        dayMiles  += r["total_miles"]
        dayOrders += len(route)

    print(f"  {day} routes: {len(routes)} | orders: {dayOrders} | total miles: {dayMiles}")
    return dayMiles, dayOrders


def printComparisonTable(orStats, cwStats):
    """Side-by-side OR-Tools vs Clarke-Wright summary table."""
    print("\nSolver Comparison")
    print("-" * 60)
    print(f"  {'Day':<6} {'OR-Tools':>12} {'CW+LS':>12} {'Diff':>10} {'OR-T routes':>12} {'CW routes':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

    totalOr = totalCw = 0
    for day in DAYS:
        orMiles  = orStats[day]["miles"]
        cwMiles  = cwStats[day]["miles"]
        diff     = orMiles - cwMiles
        pct      = (diff / cwMiles * 100) if cwMiles else 0
        totalOr += orMiles
        totalCw += cwMiles
        print(
            f"  {day:<6} {orMiles:>12,} {cwMiles:>12,} "
            f"{diff:>+9,} ({pct:+.1f}%)   "
            f"{orStats[day]['routes']:>5}       {cwStats[day]['routes']:>5}"
        )

    diffTotal = totalOr - totalCw
    pctTotal  = (diffTotal / totalCw * 100) if totalCw else 0
    print(f"  {'TOTAL':<6} {totalOr:>12,} {totalCw:>12,} {diffTotal:>+9,} ({pctTotal:+.1f}%)")
    print()
    print("  Positive diff = OR-Tools used more miles than CW (expected for a ~4% gap solver)")


def exportRouteDetails(routesByDay):
    """Write per-stop timing CSV in Table 3 format."""
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
    """Write resource summary and driver chains."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summaryDF, chainsDF = analyser.toDataFrame()
    summaryDF.to_csv(os.path.join(OUTPUT_DIR, "resource_summary.csv"), index=False)
    chainsDF.to_csv(os.path.join(OUTPUT_DIR, "driver_chains.csv"),     index=False)
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'resource_summary.csv')}")
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'driver_chains.csv')}")


def geocodeAllZips(allZips):
    """Geocode ZIP codes to (lat, lon) via pgeocode; fall back to MDS for misses."""
    import numpy as np
    import pgeocode

    nomi   = pgeocode.Nominatim(COUNTRY_CODE)
    coords = {}

    for z in allZips:
        zipStr = str(int(z)).zfill(5)
        result = nomi.query_postal_code(zipStr)
        if not pd.isna(result.latitude) and not pd.isna(result.longitude):
            coords[z] = (float(result.latitude), float(result.longitude))

    successRate = len(coords) / len(allZips) if allZips else 0
    print(f"  Geocoded {len(coords)}/{len(allZips)} ZIPs ({successRate:.0%})")

    missing = [z for z in allZips if z not in coords]
    if missing:
        print(f"  {len(missing)} ZIPs missing — filling with MDS layout...")
        mds = _mdsLayout(list(allZips))
        for z in missing:
            coords[z] = mds[z]
    return coords


def _mdsLayout(allZips):
    """MDS fallback: embed road distances into a lat/lon box centred on Boston."""
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

    coords = {}
    for i, z in enumerate(allZips):
        norm      = (pos[i] - pos.min(axis=0)) / span
        coords[z] = (42.3 + (norm[1] - 0.5) * 1.5,
                     -71.1 + (norm[0] - 0.5) * 2.0)
    return coords


def buildMap(routesByDay, zipCoords):
    """Build a Folium map with one FeatureGroup per day."""
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
        for routeIdx, route in enumerate(routesByDay[day]):
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
                         f"{len(route)} orders | drive={r['total_drive']}h | "
                         f"duty={r['total_duty']}h"),
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
                cube      = int(stop["CUBE"])
                popupHtml = (
                    f'<div style="font-family:monospace;font-size:13px;min-width:180px;">'
                    f"<b>{routeLabel}</b><br>Stop #{seq}<br>"
                    f"Order ID: {oid}<br>ZIP: {int(z)}<br>"
                    f"Cube: {cube} ft³<br>Feasible: {r['overall_feasible']}</div>"
                )
                folium.CircleMarker(
                    location=coord, radius=7, color="white", weight=2,
                    fill=True, fill_color=routeColor, fill_opacity=0.9,
                    tooltip=f"Stop {seq} · Order {oid} · ZIP {int(z)}",
                    popup=folium.Popup(popupHtml, max_width=230),
                ).add_to(dayGroup)
                folium.Marker(
                    location=coord,
                    icon=folium.DivIcon(
                        html=(f'<div style="font-size:9px;font-weight:bold;'
                              f'color:white;text-align:center;line-height:14px;">{seq}</div>'),
                        icon_size=(14, 14), icon_anchor=(7, 7),
                    ),
                ).add_to(dayGroup)

        dayGroup.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    legendHtml = (
        '<div style="position:fixed;bottom:30px;left:30px;z-index:1000;'
        'background:white;border-radius:8px;padding:12px 16px;'
        'box-shadow:0 2px 10px rgba(0,0,0,0.15);'
        'font-family:Segoe UI,sans-serif;font-size:13px;">'
        '<b style="font-size:14px;">Days</b><br><br>'
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


def main():
    parser = argparse.ArgumentParser(description="NHG OR-Tools CVRP — Q1 base case")
    parser.add_argument("--time",    type=int, default=30,
                        help="Per-day solver time limit in seconds (default: 30)")
    parser.add_argument("--compare", action="store_true",
                        help="Also run CW+LS and print a side-by-side comparison")
    parser.add_argument("--no-map",  action="store_true",
                        help="Skip geocoding and map generation")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    orders, _ = loadInputs()

    ortSolver = ORToolsSolver(timeLimitSec=args.time)
    cwSolver  = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True) if args.compare else None

    ortRoutesByDay = {}
    cwRoutesByDay  = {}
    ortStatsByDay  = {}
    cwStatsByDay   = {}

    for day in DAYS:
        dayOrders = orders[orders["DayOfWeek"] == day].copy()

        print(f"OR-Tools solving {day}  (limit={args.time}s)...")
        ortRoutesByDay[day] = ortSolver.solve(dayOrders)
        ortStatsByDay[day]  = ortSolver.getStats()

        if cwSolver:
            print(f"CW+LS solving {day}...")
            cwRoutesByDay[day] = cwSolver.solve(dayOrders)
            cwStatsByDay[day]  = cwSolver.getStats()

    print("\nOR-Tools Routes")
    print("-" * 60)
    weeklyMiles  = 0
    weeklyRoutes = 0
    weeklyOrders = 0
    for day in DAYS:
        dayMiles, dayOrds = printDayReport(day, ortRoutesByDay[day])
        weeklyMiles  += dayMiles
        weeklyRoutes += len(ortRoutesByDay[day])
        weeklyOrders += dayOrds

    print("\nWeekly Summary")
    print("-" * 60)
    print(f"  Total routes:           {weeklyRoutes}")
    print(f"  Total orders fulfilled: {weeklyOrders}")
    print(f"  Total miles:            {weeklyMiles:,}")
    print(f"  Annual miles:           {weeklyMiles * 52:,}")

    if args.compare:
        printComparisonTable(ortStatsByDay, cwStatsByDay)

    analyser = ResourceAnalyser(ortRoutesByDay)
    analyser.analyse()
    analyser.printReport()

    costModel = CostModel()
    breakdown = costModel.weeklyBreakdown(ortRoutesByDay)
    costModel.printSummary(breakdown, label="OR-Tools CVRP (Q1)")

    print("\nExporting outputs...")
    exportRouteDetails(ortRoutesByDay)
    exportResourceReport(analyser)

    if not args.no_map:
        print("\nGeocoding ZIPs...")
        allZips = {DEPOT_ZIP}
        for routes in ortRoutesByDay.values():
            for route in routes:
                for stop in route:
                    allZips.add(stop["TOZIP"])
        zipCoords = geocodeAllZips(allZips)
        print("Building map...")
        buildMap(ortRoutesByDay, zipCoords).save(MAP_FILE)
        print(f"  Saved: {MAP_FILE}")
    else:
        print("\n(Map skipped — run without --no-map to generate)")


if __name__ == "__main__":
    main()
