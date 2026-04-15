"""
VRP_OvernightRoutes.py — Overnight DOT break routing scenario for the NHG dataset.

Seeds from CW + local search, applies overnight pairings greedily,
reports the final solution with resource requirements and cost analysis.
Requires VRP_DataAnalysis.py to have been run first (data/ must exist).

Usage:
    python VRP_OvernightRoutes.py              # solver + interactive map
    python VRP_OvernightRoutes.py --no-map    # solver only (faster)

Outputs:
    outputs/overnight/route_details.csv       — per-stop timing (day-cab + overnight)
    outputs/overnight/resource_summary.csv    — headline resource metrics
    outputs/overnight/driver_chains.csv       — driver chain assignments
    outputs/overnight/routes_map_overnight.html — interactive Folium map (unless --no-map)
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
from vrp_solvers.costModel       import CostModel
from vrp_solvers.overnightSolver import (
    OvernightSolver,
    findAllOvernightCandidates,
    applyOvernightImprovements,
    ADJACENT_DAY_PAIRS,
)
from vrp_solvers.resourceAnalyser import ResourceAnalyser

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "overnight")
MAP_FILE   = os.path.join(OUTPUT_DIR, "routes_map_overnight.html")
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

OVERNIGHT_CONNECTOR_COLOR  = "#FFFFFF"
OVERNIGHT_CONNECTOR_WEIGHT = 3


def printDayReport(day, routes):
    """Print per-route details for one day; return total miles."""
    print(f"\n{day}")
    print("-" * 60)
    dayTotalMiles = 0
    for i, route in enumerate(routes, start=1):
        r = evaluateRoute(route)
        print(
            f"  Route {i}: {routeIds(route)} | "
            f"miles={r['total_miles']} | cube={r['total_cube']} | "
            f"duty={r['total_duty']}h | feasible={r['overall_feasible']}"
        )
        dayTotalMiles += r["total_miles"]
    print(f"  {day} routes: {len(routes)} | total miles: {dayTotalMiles}")
    return dayTotalMiles


def printOvernightCandidates(routesByDay):
    """Print all improving overnight pairs for each adjacent day pair."""
    for d1, d2 in ADJACENT_DAY_PAIRS:
        candidates = findAllOvernightCandidates(
            routesByDay.get(d1, []), routesByDay.get(d2, [])
        )
        print(f"\nOvernight candidates: {d1} -> {d2}")
        if not candidates:
            print("  No improving overnight pairs found.")
        else:
            for c in candidates:
                print(
                    f"  {d1} R{c['route1_idx']+1} + {d2} R{c['route2_idx']+1} | "
                    f"separate={c['separate_miles']} mi | "
                    f"overnight={c['overnight_miles']} mi | "
                    f"savings={c['savings']} mi"
                )


def printAppliedOvernights(overnightRoutes):
    """Print each applied overnight pairing."""
    print("\nApplied Overnight Routes")
    print("-" * 60)
    if not overnightRoutes:
        print("  No overnight routes applied.")
        return
    for k, ovn in enumerate(overnightRoutes, start=1):
        print(
            f"  Overnight {k}: {ovn['day1']} R{ovn['route1_idx']+1} "
            f"+ {ovn['day2']} R{ovn['route2_idx']+1} | "
            f"savings={ovn['savings']} mi | "
            f"miles={ovn['results']['total_miles']} | "
            f"feasible={ovn['results']['overall_feasible']}"
        )


def exportRouteDetails(routesByDay, overnightRoutes, usedRoutes):
    """Write per-stop timing CSV for all routes — day-cab and overnight."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    locsPath = os.path.join(DATA_DIR, "locations_clean.csv")
    locs     = pd.read_csv(locsPath) if os.path.exists(locsPath) else None

    allRows  = []
    routeNum = 1

    for ovn in overnightRoutes:
        day1Rows = detailedRouteTrace(ovn["route1"], ovn["day1"], routeNum, locs)
        day2Rows = detailedRouteTrace(ovn["route2"], ovn["day2"], routeNum, locs)
        for row in day1Rows:
            row["route_type"] = "overnight_day1"
        for row in day2Rows:
            row["route_type"] = "overnight_day2"
        if day1Rows:
            day1Rows = day1Rows[:-1]
        if day2Rows:
            day2Rows = day2Rows[1:]
        allRows.extend(day1Rows)
        allRows.extend(day2Rows)
        routeNum += 1

    for day in DAYS:
        for ridx, route in enumerate(routesByDay.get(day, [])):
            if ridx in usedRoutes.get(day, set()):
                continue
            rows = detailedRouteTrace(route, day, routeNum, locs)
            for row in rows:
                row["route_type"] = "day_cab"
            allRows.extend(rows)
            routeNum += 1

    detailDF = pd.DataFrame(allRows, columns=[
        "day", "route_number", "route_type", "stop_sequence", "order_id",
        "location", "arrival_time", "departure_time", "delivery_volume_cuft",
    ])
    detailDF.to_csv(os.path.join(OUTPUT_DIR, "route_details.csv"), index=False)
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'route_details.csv')}")


def exportResourceReport(analyser):
    """Write resource summary and driver chains. Output: outputs/overnight/"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summaryDF, chainsDF = analyser.toDataFrame()
    summaryDF.to_csv(os.path.join(OUTPUT_DIR, "resource_summary.csv"), index=False)
    chainsDF.to_csv(os.path.join(OUTPUT_DIR, "driver_chains.csv"),     index=False)
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'resource_summary.csv')}")
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'driver_chains.csv')}")


def geocodeAllZips(allZips):
    """Geocode ZIP codes to (lat, lon) via pgeocode; MDS fallback for misses."""
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


def addStopMarkers(route, routeLabel, color, r, zipCoords, featureGroup):
    """Add circle + sequence number markers for every stop in a route."""
    import folium
    for seq, stop in enumerate(route, start=1):
        z = stop["TOZIP"]
        if z not in zipCoords:
            continue
        coord = zipCoords[z]
        oid   = int(stop["ORDERID"])
        popupHtml = (
            f'<div style="font-family:monospace;font-size:13px;min-width:180px;">'
            f"<b>{routeLabel}</b><br>Stop #{seq}<br>"
            f"Order ID: {oid}<br>ZIP: {int(z)}<br>"
            f"Cube: {int(stop['CUBE'])} ft³<br>"
            f"Feasible: {r['overall_feasible']}</div>"
        )
        folium.CircleMarker(
            location=coord, radius=7, color="white", weight=2,
            fill=True, fill_color=color, fill_opacity=0.9,
            tooltip=f"Stop {seq} · Order {oid}",
            popup=folium.Popup(popupHtml, max_width=230),
        ).add_to(featureGroup)
        folium.Marker(
            location=coord,
            icon=folium.DivIcon(
                html=(f'<div style="font-size:9px;font-weight:bold;'
                      f'color:white;text-align:center;line-height:14px;">{seq}</div>'),
                icon_size=(14, 14), icon_anchor=(7, 7),
            ),
        ).add_to(featureGroup)


def buildMap(routesByDay, overnightRoutes, usedRoutes, zipCoords):
    """Build a Folium map showing overnight pairings and day-cab routes."""
    import folium
    from folium import plugins

    depotCoord = zipCoords.get(DEPOT_ZIP, (42.3, -71.1))
    m = folium.Map(location=depotCoord, zoom_start=9, tiles="CartoDB positron")

    folium.Marker(
        location=depotCoord,
        tooltip=f"<b>DEPOT</b><br>ZIP {DEPOT_ZIP}",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)

    overnightGroup = folium.FeatureGroup(name="Overnight Pairings", show=True)

    for ovnIdx, ovn in enumerate(overnightRoutes):
        pNum      = ovnIdx + 1
        d1Color   = DAY_COLORS.get(ovn["day1"], "#888")
        d2Color   = DAY_COLORS.get(ovn["day2"], "#888")
        d1Label   = f"Overnight {pNum} · {ovn['day1']} (Day 1)"
        d2Label   = f"Overnight {pNum} · {ovn['day2']} (Day 2)"
        r1        = evaluateRoute(ovn["route1"])
        r2        = evaluateRoute(ovn["route2"])
        ovnRes    = ovn["results"]

        day1Wpts = [depotCoord] + [zipCoords[s["TOZIP"]] for s in ovn["route1"]
                                   if s["TOZIP"] in zipCoords]
        folium.PolyLine(
            locations=day1Wpts, color=d1Color, weight=4, opacity=0.9,
            tooltip=(f"{d1Label} | {r1['total_miles']} mi | "
                     f"{len(ovn['route1'])} orders | drive={r1['total_drive']}h"),
        ).add_to(overnightGroup)
        addStopMarkers(ovn["route1"], d1Label, d1Color, r1, zipCoords, overnightGroup)

        if ovn["route1"] and ovn["route2"]:
            l1z = ovn["route1"][-1]["TOZIP"]
            f2z = ovn["route2"][0]["TOZIP"]
            if l1z in zipCoords and f2z in zipCoords:
                folium.PolyLine(
                    locations=[zipCoords[l1z], zipCoords[f2z]],
                    color=OVERNIGHT_CONNECTOR_COLOR,
                    weight=OVERNIGHT_CONNECTOR_WEIGHT,
                    opacity=0.7, dash_array="8 6",
                    tooltip=(f"Overnight {pNum}: transit/break leg | "
                             f"savings={ovn['savings']} mi | "
                             f"total={ovnRes['total_miles']} mi"),
                ).add_to(overnightGroup)

        day2Wpts = [zipCoords[s["TOZIP"]] for s in ovn["route2"]
                    if s["TOZIP"] in zipCoords] + [depotCoord]
        folium.PolyLine(
            locations=day2Wpts, color=d2Color, weight=4, opacity=0.9,
            tooltip=(f"{d2Label} | {r2['total_miles']} mi | "
                     f"{len(ovn['route2'])} orders | drive={r2['total_drive']}h"),
        ).add_to(overnightGroup)
        addStopMarkers(ovn["route2"], d2Label, d2Color, r2, zipCoords, overnightGroup)

    overnightGroup.add_to(m)

    for day in DAYS:
        dayGroup = folium.FeatureGroup(name=f"{day} (day-cab)", show=True)
        for routeIdx, route in enumerate(routesByDay.get(day, [])):
            if routeIdx in usedRoutes.get(day, set()):
                continue
            routeColor = ROUTE_PALETTE[routeIdx % len(ROUTE_PALETTE)]
            r          = evaluateRoute(route)
            routeLabel = f"{day} · Route {routeIdx + 1}"
            waypoints  = ([depotCoord]
                          + [zipCoords[s["TOZIP"]] for s in route if s["TOZIP"] in zipCoords]
                          + [depotCoord])
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
            addStopMarkers(route, routeLabel, routeColor, r, zipCoords, dayGroup)
        dayGroup.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main():
    parser = argparse.ArgumentParser(description="NHG Overnight Routes")
    parser.add_argument("--no-map", action="store_true",
                        help="Skip geocoding and map generation")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    orders, _ = loadInputs()

    solver = OvernightSolver(ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True))
    routesByDay, overnightRoutes, usedRoutes = solver.solve(orders)

    weeklyTotalMiles  = 0
    weeklyTotalRoutes = 0
    for day in DAYS:
        weeklyTotalMiles  += printDayReport(day, routesByDay[day])
        weeklyTotalRoutes += len(routesByDay[day])

    print("\nWeekly Summary (before overnight)")
    print("-" * 60)
    print(f"  Total routes: {weeklyTotalRoutes}")
    print(f"  Total miles:  {weeklyTotalMiles:,}")

    printOvernightCandidates(routesByDay)
    printAppliedOvernights(overnightRoutes)

    stats = solver.getStats()
    print("\nFinal Summary with Overnight")
    print("-" * 60)
    print(f"  Total routes:    {stats['routes']}")
    print(f"  Total miles:     {stats['miles']:,}")
    print(f"  Annual miles:    {stats['miles'] * 52:,}")
    print(f"  Overnight pairs: {stats['overnight_pairs']}")

    analyser = ResourceAnalyser(routesByDay, overnightPairings=overnightRoutes)
    analyser.analyse()
    analyser.printReport()

    costModel    = CostModel()
    breakdown    = costModel.weeklyBreakdown(routesByDay, overnightPairings=overnightRoutes)
    costModel.printSummary(breakdown, label="Overnight Routes (CW + LS)")
    overnightSum = costModel.overnightSummary(routesByDay, overnightRoutes)
    costModel.printOvernightSummary(overnightSum)

    print("\nExporting outputs...")
    exportRouteDetails(routesByDay, overnightRoutes, usedRoutes)
    exportResourceReport(analyser)

    if not args.no_map:
        print("\nGeocoding ZIPs...")
        allZips = {DEPOT_ZIP}
        for routes in routesByDay.values():
            for route in routes:
                for stop in route:
                    allZips.add(stop["TOZIP"])
        for ovn in overnightRoutes:
            for stop in ovn["route1"] + ovn["route2"]:
                allZips.add(stop["TOZIP"])
        zipCoords = geocodeAllZips(allZips)
        print("Building map...")
        buildMap(routesByDay, overnightRoutes, usedRoutes, zipCoords).save(MAP_FILE)
        print(f"  Saved: {MAP_FILE}")
    else:
        print("\n(Map skipped — run without --no-map to generate)")


if __name__ == "__main__":
    main()