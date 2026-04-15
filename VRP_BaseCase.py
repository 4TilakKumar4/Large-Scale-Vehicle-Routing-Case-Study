"""
VRP_BaseCase.py — Clarke-Wright + 2-opt + or-opt on the NHG dataset.

Requires VRP_DataAnalysis.py to have been run first (data/ must exist).

Usage:
    python VRP_BaseCase.py              # solver + interactive map
    python VRP_BaseCase.py --no-map    # solver only (faster, skips geocoding)

Outputs:
    outputs/base_case/route_details.csv      — per-stop timing in Table 3 format
    outputs/base_case/resource_summary.csv   — headline resource metrics
    outputs/base_case/driver_chains.csv      — driver chain assignments
    outputs/base_case/routes_map_baseCase.html — interactive Folium map (unless --no-map)
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
from vrp_solvers.clarkeWright    import ClarkeWrightSolver
from vrp_solvers.resourceAnalyser import ResourceAnalyser
from vrp_solvers.costModel        import CostModel

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "base_case")
MAP_FILE   = os.path.join(OUTPUT_DIR, "routes_map_baseCase.html")
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


def printDayReport(day, routes):
    """Print per-route details for one day; return (total_miles, total_orders)."""
    print(f"\n{day}")
    print("-" * 60)
    dayTotalMiles  = 0
    dayTotalOrders = 0

    for i, route in enumerate(routes, start=1):
        r          = evaluateRoute(route)
        orderCount = len(route)
        print(
            f"  Route {i}: {routeIds(route)} | "
            f"orders={orderCount} | "
            f"miles={r['total_miles']} | cube={r['total_cube']} | "
            f"drive={r['total_drive']}h | duty={r['total_duty']}h | "
            f"cap={r['capacity_feasible']} dot={r['dot_feasible']} "
            f"win={r['window_feasible']} | feasible={r['overall_feasible']}"
        )
        dayTotalMiles  += r["total_miles"]
        dayTotalOrders += orderCount

    print(f"  {day} routes: {len(routes)} | orders: {dayTotalOrders} | total miles: {dayTotalMiles}")
    return dayTotalMiles, dayTotalOrders


def exportRouteDetails(routesByDay):
    """Write per-stop timing CSV in Table 3 format. Output: outputs/base_case/route_details.csv"""
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
    """Write resource summary and driver chains. Output: outputs/base_case/"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summaryDF, chainsDF = analyser.toDataFrame()
    summaryDF.to_csv(os.path.join(OUTPUT_DIR, "resource_summary.csv"), index=False)
    chainsDF.to_csv(os.path.join(OUTPUT_DIR, "driver_chains.csv"),     index=False)
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'resource_summary.csv')}")
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'driver_chains.csv')}")


def geocodeAllZips(allZips):
    """Geocode ZIP codes to (lat, lon) via pgeocode; fall back to MDS layout for misses."""
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

    pos    = MDS(n_components=2, dissimilarity="precomputed",
                 random_state=42, normalized_stress="auto").fit_transform(D)
    span   = pos.max(axis=0) - pos.min(axis=0)
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
                coord = zipCoords[z]
                oid   = int(stop["ORDERID"])
                cube  = int(stop["CUBE"])
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
    parser = argparse.ArgumentParser(description="NHG Base Case — CW + local search")
    parser.add_argument("--no-map", action="store_true",
                        help="Skip geocoding and map generation (faster for solver iteration)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    orders, _ = loadInputs()
    solver    = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)

    routesByDay       = {}
    weeklyTotalMiles  = 0
    weeklyTotalRoutes = 0
    weeklyTotalOrders = 0

    for day in DAYS:
        print(f"Building routes for {day}...")
        dayOrders        = orders[orders["DayOfWeek"] == day].copy()
        routesByDay[day] = solver.solve(dayOrders)

    for day in DAYS:
        dayMiles, dayOrds = printDayReport(day, routesByDay[day])
        weeklyTotalMiles  += dayMiles
        weeklyTotalRoutes += len(routesByDay[day])
        weeklyTotalOrders += dayOrds

    print("\nWeekly Summary")
    print("-" * 60)
    print(f"  Total routes:           {weeklyTotalRoutes}")
    print(f"  Total orders fulfilled: {weeklyTotalOrders}")
    print(f"  Total miles:            {weeklyTotalMiles:,}")
    print(f"  Annual miles:           {weeklyTotalMiles * 52:,}")

    analyser = ResourceAnalyser(routesByDay)
    analyser.analyse()
    analyser.printReport()

    costModel = CostModel()
    breakdown = costModel.weeklyBreakdown(routesByDay)
    costModel.printSummary(breakdown, label="Base Case (CW + LS)")

    print("\nExporting outputs...")
    exportRouteDetails(routesByDay)
    exportResourceReport(analyser)

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