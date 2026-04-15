"""
VRP_OvernightRoutes_Map.py — Overnight routing scenario with interactive Folium map.

Runs CW + local search, applies overnight pairings, then renders a map that visually
distinguishes overnight two-day routes from regular day-cab routes.

Overnight pairings are shown with:
  - A solid line for each day's delivery segment (day-1 color, day-2 color)
  - A thick dashed connector between the last day-1 stop and the first day-2 stop,
    representing the overnight transit/break leg

Day-cab routes are shown in the standard animated AntPath style.

Requires VRP_DataAnalysis.py to have been run first (data/ must exist).
Outputs:
  outputs/overnight/routes_map_overnight.html  — interactive Folium map
  outputs/overnight/route_details.csv           — per-stop timing for all routes
  outputs/overnight/resource_summary.csv        — headline resource metrics
  outputs/overnight/driver_chains.csv           — driver chain assignments
"""

import os

import folium
import numpy as np
import pandas as pd
import pgeocode
from folium import plugins
from sklearn.manifold import MDS

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

OVERNIGHT_CONNECTOR_COLOR = "#222222"   # dark connector for overnight transit leg
OVERNIGHT_CONNECTOR_WEIGHT = 4


# Geocoding helpers (shared with BaseCase_Map)

def geocodeAllZips(allZips):
    """Geocode ZIP codes to (lat, lon) via pgeocode; fall back to MDS if needed."""
    nomi   = pgeocode.Nominatim(COUNTRY_CODE)
    coords = {}

    for z in allZips:
        zipStr = str(int(z)).zfill(5)
        result = nomi.query_postal_code(zipStr)
        if not pd.isna(result.latitude) and not pd.isna(result.longitude):
            coords[z] = (float(result.latitude), float(result.longitude))

    successRate = len(coords) / len(allZips) if allZips else 0
    print(f"Geocoded {len(coords)}/{len(allZips)} ZIPs via pgeocode "
          f"(success rate: {successRate:.0%})")

    missing = [z for z in allZips if z not in coords]
    if missing:
        print(f"  {len(missing)} ZIPs not found — filling with MDS layout...")
        mdsCoords = mdsLayout(list(allZips))
        for z in missing:
            coords[z] = mdsCoords[z]

    return coords


def mdsLayout(allZips):
    """Fallback geocoder: MDS on road distances scaled to a lat/lon box centred on Boston."""
    n = len(allZips)
    D = np.zeros((n, n))

    for i, za in enumerate(allZips):
        for j, zb in enumerate(allZips):
            if i != j:
                try:
                    D[i, j] = float(getDistance(za, zb))
                except Exception:
                    D[i, j] = 999

    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=42, normalized_stress="auto")
    pos = mds.fit_transform(D)

    posMin = pos.min(axis=0)
    posMax = pos.max(axis=0)
    span   = posMax - posMin
    span[span == 0] = 1

    centreLat, centreLon = 42.3, -71.1
    scaleLat,  scaleLon  = 1.5,   2.0

    coords = {}
    for i, z in enumerate(allZips):
        norm      = (pos[i] - posMin) / span
        lat       = centreLat + (norm[1] - 0.5) * scaleLat
        lon       = centreLon + (norm[0] - 0.5) * scaleLon
        coords[z] = (lat, lon)

    return coords


# Map helpers

def addStopMarkers(route, routeLabel, routeColor, routeResult, zipCoords, group):
    """Add circle markers and sequence number labels for every stop on a route."""
    for seq, stop in enumerate(route, start=1):
        z = stop["TOZIP"]
        if z not in zipCoords:
            continue

        coord = zipCoords[z]
        oid   = int(stop["ORDERID"])
        cube  = int(stop["CUBE"])

        popupHtml = f"""
        <div style="font-family: monospace; font-size: 13px; min-width: 180px;">
            <b>{routeLabel}</b><br>
            Stop #{seq}<br>
            Order ID: {oid}<br>
            ZIP: {z}<br>
            Cube: {cube} ft³<br>
            Feasible: {routeResult['overall_feasible']}
        </div>
        """

        folium.CircleMarker(
            location=coord,
            radius=7,
            color="white",
            weight=2,
            fill=True,
            fill_color=routeColor,
            fill_opacity=0.9,
            tooltip=f"Stop {seq} · Order {oid} · ZIP {z}",
            popup=folium.Popup(popupHtml, max_width=230),
        ).add_to(group)

        folium.Marker(
            location=coord,
            icon=folium.DivIcon(
                html=(f'<div style="font-size:9px;font-weight:bold;'
                      f'color:white;text-align:center;line-height:14px;">'
                      f'{seq}</div>'),
                icon_size=(14, 14),
                icon_anchor=(7, 7),
            ),
        ).add_to(group)


def buildMap(routesByDay, overnightRoutes, usedRoutes, zipCoords):
    """
    Build a Folium map with:
      - One FeatureGroup per day for day-cab routes (animated AntPath)
      - One FeatureGroup for overnight pairings, showing day-1 segment,
        overnight connector leg, and day-2 segment as distinct visual elements
    """
    depotCoord = zipCoords.get(DEPOT_ZIP, (42.3, -71.1))

    m = folium.Map(location=depotCoord, zoom_start=9, tiles="CartoDB positron")

    folium.Marker(
        location=depotCoord,
        tooltip=f"<b>DEPOT</b><br>ZIP {DEPOT_ZIP}",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)

    # --- Overnight pairings FeatureGroup ---
    overnightGroup = folium.FeatureGroup(name="Overnight Pairings", show=True)

    for ovnIdx, ovn in enumerate(overnightRoutes):
        pairingNum    = ovnIdx + 1
        day1Color     = DAY_COLORS.get(ovn["day1"], "#888888")
        day2Color     = DAY_COLORS.get(ovn["day2"], "#888888")
        day1Label     = f"Overnight {pairingNum} · {ovn['day1']} (Day 1)"
        day2Label     = f"Overnight {pairingNum} · {ovn['day2']} (Day 2)"
        r1            = evaluateRoute(ovn["route1"])
        r2            = evaluateRoute(ovn["route2"])
        overnightRes  = ovn["results"]

        # Day-1 segment: depot → stops
        day1Waypoints = [depotCoord]
        for stop in ovn["route1"]:
            z = stop["TOZIP"]
            if z in zipCoords:
                day1Waypoints.append(zipCoords[z])

        folium.PolyLine(
            locations=day1Waypoints,
            color=day1Color,
            weight=4,
            opacity=0.9,
            tooltip=(
                f"{day1Label} | {r1['total_miles']} mi | "
                f"{len(ovn['route1'])} orders | drive={r1['total_drive']}h"
            ),
        ).add_to(overnightGroup)

        addStopMarkers(ovn["route1"], day1Label, day1Color, r1, zipCoords, overnightGroup)

        # Overnight connector: last day-1 stop → first day-2 stop (dashed)
        if ovn["route1"] and ovn["route2"]:
            lastDay1Zip  = ovn["route1"][-1]["TOZIP"]
            firstDay2Zip = ovn["route2"][0]["TOZIP"]
            if lastDay1Zip in zipCoords and firstDay2Zip in zipCoords:
                connectorPoints = [zipCoords[lastDay1Zip], zipCoords[firstDay2Zip]]
                folium.PolyLine(
                    locations=connectorPoints,
                    color=OVERNIGHT_CONNECTOR_COLOR,
                    weight=OVERNIGHT_CONNECTOR_WEIGHT,
                    opacity=0.7,
                    dash_array="8 6",
                    tooltip=(
                        f"Overnight {pairingNum}: overnight transit/break leg | "
                        f"savings={ovn['savings']} mi | "
                        f"total overnight miles={overnightRes['total_miles']}"
                    ),
                ).add_to(overnightGroup)

        # Day-2 segment: stops → depot (no depot dispatch row on map)
        day2Waypoints = []
        for stop in ovn["route2"]:
            z = stop["TOZIP"]
            if z in zipCoords:
                day2Waypoints.append(zipCoords[z])
        day2Waypoints.append(depotCoord)

        folium.PolyLine(
            locations=day2Waypoints,
            color=day2Color,
            weight=4,
            opacity=0.9,
            tooltip=(
                f"{day2Label} | {r2['total_miles']} mi | "
                f"{len(ovn['route2'])} orders | drive={r2['total_drive']}h"
            ),
        ).add_to(overnightGroup)

        addStopMarkers(ovn["route2"], day2Label, day2Color, r2, zipCoords, overnightGroup)

    overnightGroup.add_to(m)

    # --- Day-cab FeatureGroups (one per day) ---
    for day in DAYS:
        dayGroup = folium.FeatureGroup(name=f"{day} (day-cab)", show=True)
        routes   = routesByDay.get(day, [])

        for routeIdx, route in enumerate(routes):
            if routeIdx in usedRoutes.get(day, set()):
                continue   # rendered in overnight group above

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
                locations=waypoints,
                color=routeColor,
                weight=3,
                opacity=0.85,
                tooltip=(
                    f"{routeLabel} | {r['total_miles']} mi | "
                    f"{len(route)} orders | drive={r['total_drive']}h | "
                    f"duty={r['total_duty']}h"
                ),
            ).add_to(dayGroup)

            plugins.AntPath(
                locations=waypoints,
                color=routeColor,
                weight=3,
                opacity=0.5,
                delay=1200,
                dash_array=[10, 40],
            ).add_to(dayGroup)

            addStopMarkers(route, routeLabel, routeColor, r, zipCoords, dayGroup)

        dayGroup.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    legendHtml = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: white; border-radius: 8px; padding: 12px 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        font-family: 'Segoe UI', sans-serif; font-size: 13px;
    ">
        <b style="font-size:14px;">Legend</b><br><br>
    """
    for day, color in DAY_COLORS.items():
        legendHtml += (
            f'<div style="display:flex;align-items:center;margin-bottom:5px;">'
            f'<div style="width:14px;height:14px;border-radius:50%;'
            f'background:{color};margin-right:8px;flex-shrink:0;"></div>'
            f'{day}</div>'
        )
    legendHtml += (
        f'<div style="display:flex;align-items:center;margin-top:8px;">'
        f'<div style="width:28px;height:4px;background:{OVERNIGHT_CONNECTOR_COLOR};'
        f'border-top:2px dashed {OVERNIGHT_CONNECTOR_COLOR};margin-right:8px;'
        f'flex-shrink:0;"></div>'
        f'Overnight transit</div>'
    )
    legendHtml += "</div>"
    m.get_root().html.add_child(folium.Element(legendHtml))

    return m


 
# CSV export helpers
 

def exportRouteDetails(routesByDay, overnightRoutes, usedRoutes):
    """Write per-stop timing CSV for all routes. Output: outputs/overnight/route_details.csv"""
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

        # Drop closing depot row from day1 and opening depot row from day2
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

    outPath = os.path.join(OUTPUT_DIR, "route_details.csv")
    detailDF.to_csv(outPath, index=False)
    print(f"  Saved: {outPath}")


def exportResourceReport(analyser):
    """Write resource summary and driver chains. Output: outputs/overnight/"""
    summaryDF, chainsDF = analyser.toDataFrame()
    summaryPath = os.path.join(OUTPUT_DIR, "resource_summary.csv")
    chainsPath  = os.path.join(OUTPUT_DIR, "driver_chains.csv")
    summaryDF.to_csv(summaryPath, index=False)
    chainsDF.to_csv(chainsPath,   index=False)
    print(f"  Saved: {summaryPath}")
    print(f"  Saved: {chainsPath}")


# Main

def main():
    """Build overnight routes, render map, and export all outputs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    orders, _ = loadInputs()

    solver = OvernightSolver(ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True))
    routesByDay, overnightRoutes, usedRoutes = solver.solve(orders)

    stats = solver.getStats()
    print("\nFinal Summary with Overnight")
    print("-" * 60)
    print("Total routes:    ", stats["routes"])
    print("Total miles:     ", stats["miles"])
    print("Overnight pairs: ", stats["overnight_pairs"])

    analyser = ResourceAnalyser(routesByDay, overnightPairings=overnightRoutes)
    analyser.analyse()
    analyser.printReport()

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
    m = buildMap(routesByDay, overnightRoutes, usedRoutes, zipCoords)
    m.save(MAP_FILE)
    print(f"  Saved: {MAP_FILE}")

    print("\nExporting CSVs...")
    exportRouteDetails(routesByDay, overnightRoutes, usedRoutes)
    exportResourceReport(analyser)


if __name__ == "__main__":
    main()