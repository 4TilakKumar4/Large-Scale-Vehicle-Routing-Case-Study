"""
VRP_MixedFleet_Map.py — Mixed-fleet routing with interactive Folium map.

Sub-problem 2: Vans (3,200 ft³) and Straight Trucks (1,400 ft³).
Runs MixedFleetSolver per day, then renders an interactive Folium map that
distinguishes Van routes (solid lines, circular stops) from ST routes
(dashed lines, square stops). One FeatureGroup per day×fleet combination.

Requires VRP_DataAnalysis.py to have been run first (data/ must exist).
Outputs:
  outputs/mixed_fleet/routes_map_mixed_fleet.html — interactive Folium map
  outputs/mixed_fleet/route_details.csv           — per-stop timing + vehicle type
  outputs/mixed_fleet/resource_summary.csv        — headline resource metrics
  outputs/mixed_fleet/driver_chains.csv           — driver chain assignments
  outputs/mixed_fleet/fleet_analysis.png          — daily miles + utilisation plot
"""

import os

import matplotlib
matplotlib.use("Agg")
import colorsys
import matplotlib.pyplot as plt
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
    ST_CAPACITY,
    VAN_CAPACITY,
    detailedRouteTrace,
    evaluateMixedRoute,
    getDistance,
    loadInputs,
    routeIds,
)
from vrp_solvers.mixedFleetSolver import MixedFleetSolver, _getMiles
from vrp_solvers.resourceAnalyser import ResourceAnalyser

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "mixed_fleet")

COUNTRY_CODE = "US"

# Van colours — solid, saturated, one per day
VAN_DAY_COLORS = {
    "Mon": "#E63946",
    "Tue": "#F4A261",
    "Wed": "#2A9D8F",
    "Thu": "#457B9D",
    "Fri": "#9B5DE5",
}

# ST colours — lighter tint of each day's Van colour
ST_DAY_COLORS = {
    "Mon": "#FF8A80",
    "Tue": "#FFD180",
    "Wed": "#82B1FF",
    "Thu": "#80CBC4",
    "Fri": "#EA80FC",
}

ROUTE_PALETTE = [
    "#E63946", "#F4A261", "#2A9D8F", "#457B9D",
    "#9B5DE5", "#F72585", "#4CC9F0", "#06D6A0",
]

DAY_COLORS_PLOT = {
    "Mon": {"van": "#e53935", "st": "#ff8a80"},
    "Tue": {"van": "#fb8c00", "st": "#ffd180"},
    "Wed": {"van": "#1e88e5", "st": "#82b1ff"},
    "Thu": {"van": "#8e24aa", "st": "#ea80fc"},
    "Fri": {"van": "#43a047", "st": "#b9f6ca"},
}


# ---------------------------------------------------------------------------
# Geocoding helpers (shared with base case map)
# ---------------------------------------------------------------------------

def geocodeAllZips(allZips):
    """Geocode ZIP codes via pgeocode; fill missing with MDS layout."""
    nomi   = pgeocode.Nominatim(COUNTRY_CODE)
    coords = {}

    for z in allZips:
        zipStr = str(int(z)).zfill(5)
        result = nomi.query_postal_code(zipStr)
        if not pd.isna(result.latitude) and not pd.isna(result.longitude):
            coords[z] = (float(result.latitude), float(result.longitude))

    successRate = len(coords) / len(allZips) if allZips else 0
    print(f"  Geocoded {len(coords)}/{len(allZips)} ZIPs "
          f"(success rate: {successRate:.0%})")

    missing = [z for z in allZips if z not in coords]
    if missing:
        print(f"  {len(missing)} ZIPs not found — filling with MDS layout...")
        mdsCoords = _mdsLayout(list(allZips))
        for z in missing:
            coords[z] = mdsCoords[z]

    return coords


def _mdsLayout(allZips):
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
        coords[z] = (
            centreLat + (norm[1] - 0.5) * scaleLat,
            centreLon + (norm[0] - 0.5) * scaleLon,
        )
    return coords


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def buildMap(vanByDay, stByDay, zipCoords):
    """
    Build a Folium map with two FeatureGroups per day — one for Van routes
    and one for ST routes. Van routes use solid lines and circular markers;
    ST routes use dashed lines and square markers.
    """
    depotCoord = zipCoords.get(DEPOT_ZIP, (42.3, -71.1))

    m = folium.Map(location=depotCoord, zoom_start=8, tiles="CartoDB positron")

    # Depot marker
    folium.Marker(
        location=depotCoord,
        tooltip="<b>DEPOT</b><br>Wilmington, MA",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)

    for day in DAYS:
        vanColor = VAN_DAY_COLORS[day]
        stColor  = ST_DAY_COLORS[day]

        # --- Van FeatureGroup ---
        vanGroup  = folium.FeatureGroup(name=f"{day} — Van", show=True)
        vanRoutes = vanByDay.get(day, [])

        for routeIdx, route in enumerate(vanRoutes):
            routeColor = ROUTE_PALETTE[routeIdx % len(ROUTE_PALETTE)]
            r          = evaluateMixedRoute(route, "van")
            routeLabel = f"{day} · Van R{routeIdx + 1}"

            waypoints = [depotCoord]
            for stop in route:
                z = stop["TOZIP"]
                if z in zipCoords:
                    waypoints.append(zipCoords[z])
            waypoints.append(depotCoord)

            # Solid polyline for Van
            folium.PolyLine(
                locations=waypoints,
                color=routeColor,
                weight=3,
                opacity=0.9,
                tooltip=(
                    f"{routeLabel} | {r['total_miles']} mi | "
                    f"{len(route)} orders | cube={r['total_cube']}/{VAN_CAPACITY} ft³ | "
                    f"drive={r['total_drive']}h | duty={r['total_duty']}h"
                ),
            ).add_to(vanGroup)

            # Animated direction arrows
            plugins.AntPath(
                locations=waypoints,
                color=routeColor,
                weight=3,
                opacity=0.5,
                delay=1200,
                dash_array=[10, 40],
            ).add_to(vanGroup)

            # Circular stop markers for Van
            for seq, stop in enumerate(route, start=1):
                z = stop["TOZIP"]
                if z not in zipCoords:
                    continue
                coord = zipCoords[z]
                oid   = int(stop["ORDERID"])
                cube  = int(stop["CUBE"])

                popupHtml = f"""
                <div style="font-family: monospace; font-size: 13px; min-width: 190px;">
                    <b>{routeLabel}</b><br>
                    Stop #{seq}<br>
                    Order ID: {oid}<br>
                    ZIP: {int(z)}<br>
                    Cube: {cube} ft³<br>
                    Vehicle: Van<br>
                    Feasible: {r['overall_feasible']}
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
                    tooltip=f"Van Stop {seq} · Order {oid}",
                    popup=folium.Popup(popupHtml, max_width=230),
                ).add_to(vanGroup)

                folium.Marker(
                    location=coord,
                    icon=folium.DivIcon(
                        html=(f'<div style="font-size:9px;font-weight:bold;'
                              f'color:white;text-align:center;line-height:14px;">'
                              f'{seq}</div>'),
                        icon_size=(14, 14),
                        icon_anchor=(7, 7),
                    ),
                ).add_to(vanGroup)

        vanGroup.add_to(m)

        # --- ST FeatureGroup ---
        stGroup  = folium.FeatureGroup(name=f"{day} — ST", show=True)
        stRoutes = stByDay.get(day, [])

        for routeIdx, route in enumerate(stRoutes):
            # Use ST colour palette — offset index to distinguish from Van routes
            routeColor = ST_DAY_COLORS[day]
            r          = evaluateMixedRoute(route, "st")
            routeLabel = f"{day} · ST R{routeIdx + 1}"

            waypoints = [depotCoord]
            for stop in route:
                z = stop["TOZIP"]
                if z in zipCoords:
                    waypoints.append(zipCoords[z])
            waypoints.append(depotCoord)

            # Dashed polyline for ST — visually distinct from Van solid lines
            folium.PolyLine(
                locations=waypoints,
                color=routeColor,
                weight=3,
                opacity=0.9,
                dash_array="8 6",
                tooltip=(
                    f"{routeLabel} | {r['total_miles']} mi | "
                    f"{len(route)} orders | cube={r['total_cube']}/{ST_CAPACITY} ft³ | "
                    f"drive={r['total_drive']}h | duty={r['total_duty']}h"
                ),
            ).add_to(stGroup)

            # Square stop markers for ST — distinguished from Van circles
            for seq, stop in enumerate(route, start=1):
                z = stop["TOZIP"]
                if z not in zipCoords:
                    continue
                coord = zipCoords[z]
                oid   = int(stop["ORDERID"])
                cube  = int(stop["CUBE"])
                stReq = str(stop.get("straight_truck_required", "no")).lower() == "yes"

                popupHtml = f"""
                <div style="font-family: monospace; font-size: 13px; min-width: 190px;">
                    <b>{routeLabel}</b><br>
                    Stop #{seq}<br>
                    Order ID: {oid}<br>
                    ZIP: {int(z)}<br>
                    Cube: {cube} ft³<br>
                    Vehicle: Straight Truck<br>
                    ST Required: {'Yes' if stReq else 'No (flexible)'}<br>
                    Feasible: {r['overall_feasible']}
                </div>
                """

                # Square DivIcon for ST stops
                folium.Marker(
                    location=coord,
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="width:14px;height:14px;'
                            f'background:{routeColor};border:2px solid white;'
                            f'font-size:9px;font-weight:bold;color:white;'
                            f'text-align:center;line-height:14px;">'
                            f'{seq}</div>'
                        ),
                        icon_size=(14, 14),
                        icon_anchor=(7, 7),
                    ),
                    tooltip=f"ST Stop {seq} · Order {oid}",
                    popup=folium.Popup(popupHtml, max_width=230),
                ).add_to(stGroup)

        stGroup.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    legendHtml = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: white; border-radius: 8px; padding: 14px 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        font-family: 'Segoe UI', sans-serif; font-size: 13px; min-width: 200px;
    ">
        <b style="font-size:14px;">Mixed Fleet</b><br><br>
        <b>Van Routes</b> (solid line, ● stop)<br>
        <b>ST Routes</b> (dashed line, ■ stop)<br><br>
        <b>Days</b><br><br>
    """
    for day, vanC in VAN_DAY_COLORS.items():
        stC = ST_DAY_COLORS[day]
        legendHtml += (
            f'<div style="display:flex;align-items:center;margin-bottom:5px;">'
            f'<div style="width:14px;height:14px;border-radius:50%;'
            f'background:{vanC};margin-right:6px;flex-shrink:0;"></div>'
            f'<div style="width:14px;height:14px;'
            f'background:{stC};margin-right:8px;flex-shrink:0;"></div>'
            f'{day}</div>'
        )
    legendHtml += "</div>"
    m.get_root().html.add_child(folium.Element(legendHtml))

    return m


# ---------------------------------------------------------------------------
# Exports (same as VRP_MixedFleet.py)
# ---------------------------------------------------------------------------

def exportRouteDetails(vanByDay, stByDay):
    locsPath = os.path.join(DATA_DIR, "locations_clean.csv")
    locs     = pd.read_csv(locsPath) if os.path.exists(locsPath) else None
    allRows  = []
    for day in DAYS:
        routeNum = 1
        for route in vanByDay.get(day, []):
            rows = detailedRouteTrace(route, day, routeNum, locs)
            for row in rows:
                row["vehicle_type"] = "Van"
            allRows.extend(rows)
            routeNum += 1
        for route in stByDay.get(day, []):
            rows = detailedRouteTrace(route, day, routeNum, locs)
            for row in rows:
                row["vehicle_type"] = "StraightTruck"
            allRows.extend(rows)
            routeNum += 1

    detailDF = pd.DataFrame(allRows, columns=[
        "day", "route_number", "vehicle_type", "stop_sequence", "order_id",
        "location", "arrival_time", "departure_time", "delivery_volume_cuft",
    ])
    outPath = os.path.join(OUTPUT_DIR, "route_details.csv")
    detailDF.to_csv(outPath, index=False)
    print(f"  Saved: {outPath}")


def exportResourceReport(analyser):
    summaryDF, chainsDF = analyser.toDataFrame()
    summaryDF.to_csv(os.path.join(OUTPUT_DIR, "resource_summary.csv"), index=False)
    chainsDF.to_csv(os.path.join(OUTPUT_DIR, "driver_chains.csv"),   index=False)
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'resource_summary.csv')}")
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'driver_chains.csv')}")


def plotFleetAnalysis(vanByDay, stByDay):
    vanMi   = [sum(_getMiles(r, "van") for r in vanByDay.get(d, [])) for d in DAYS]
    stMi    = [sum(_getMiles(r, "st")  for r in stByDay.get(d, []))  for d in DAYS]
    vanUtil = [
        sum(float(s["CUBE"]) for r in vanByDay.get(d, []) for s in r)
        / max(1, len(vanByDay.get(d, [])) * VAN_CAPACITY) * 100
        for d in DAYS
    ]
    stUtil = [
        sum(float(s["CUBE"]) for r in stByDay.get(d, []) for s in r)
        / max(1, len(stByDay.get(d, [])) * ST_CAPACITY) * 100
        for d in DAYS
    ]
    nVan = sum(len(vanByDay.get(d, [])) for d in DAYS)
    nSt  = sum(len(stByDay.get(d, []))  for d in DAYS)

    vanColor = "#4db8e8"
    stColor  = "#f0a030"
    x        = np.arange(len(DAYS))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f1117")

    ax = axes[0]
    ax.set_facecolor("#0f1117")
    ax.bar(x, vanMi, 0.5, label=f"Van ({nVan})",
           color=vanColor, edgecolor="white", lw=0.5, zorder=3)
    ax.bar(x, stMi, 0.5, bottom=vanMi, label=f"ST ({nSt})",
           color=stColor, edgecolor="white", lw=0.5, zorder=3)
    for i, (v, s) in enumerate(zip(vanMi, stMi)):
        ax.text(i, v + s + 15, f"{v + s:.0f}",
                ha="center", color="white", fontsize=9.5, fontweight="bold")
        if v > 50:
            ax.text(i, v / 2, f"{v:.0f}",
                    ha="center", color="white", fontsize=8)
        if s > 50:
            ax.text(i, v + s / 2, f"{s:.0f}",
                    ha="center", color="#0f1117", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(DAYS)
    ax.set_ylabel("Miles", color="white")
    ax.set_title("Daily Miles — Van vs Straight Truck",
                 color="white", fontweight="bold")
    ax.legend(facecolor="#1e2130", labelcolor="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.yaxis.grid(True, color="#333", ls="--", zorder=0)
    for sp in ax.spines.values():
        sp.set_color("#444")

    ax2 = axes[1]
    ax2.set_facecolor("#0f1117")
    ax2.bar(x - 0.13, vanUtil, 0.26, label="Van %",
            color=vanColor, edgecolor="white", lw=0.5, zorder=3)
    ax2.bar(x + 0.13, stUtil,  0.26, label="ST %",
            color=stColor,  edgecolor="white", lw=0.5, zorder=3)
    ax2.axhline(100, color="#e05252", ls="--", lw=1.5, label="100% cap", zorder=5)
    for i, (v, s) in enumerate(zip(vanUtil, stUtil)):
        ax2.text(i - 0.13, v + 1.5, f"{v:.0f}%",
                 ha="center", color=vanColor, fontsize=8, fontweight="bold")
        ax2.text(i + 0.13, s + 1.5, f"{s:.0f}%",
                 ha="center", color=stColor,  fontsize=8, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(DAYS)
    ax2.set_ylim(0, 118)
    ax2.set_title("Avg Capacity Utilisation %", color="white", fontweight="bold")
    ax2.set_ylabel("Utilisation %", color="white")
    ax2.legend(facecolor="#1e2130", labelcolor="white", fontsize=9)
    ax2.tick_params(colors="white")
    ax2.yaxis.grid(True, color="#333", ls="--", zorder=0)
    for sp in ax2.spines.values():
        sp.set_color("#444")

    fig.suptitle("Mixed Fleet — Fleet Analysis", color="white",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    outPath = os.path.join(OUTPUT_DIR, "fleet_analysis.png")
    plt.savefig(outPath, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  Saved: {outPath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run mixed fleet solver, render Folium map, export all outputs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    orders, _ = loadInputs()
    solver    = MixedFleetSolver()

    vanByDay = {}
    stByDay  = {}
    weeklyMiles  = 0
    weeklyRoutes = 0

    for day in DAYS:
        print(f"Solving {day}...", end=" ", flush=True)
        dayOrders = orders[orders["DayOfWeek"] == day].copy()
        solver.solve(dayOrders)
        stats            = solver.getStats()
        vanByDay[day]    = solver.getVanRoutes()
        stByDay[day]     = solver.getStRoutes()
        weeklyMiles     += stats["miles"]
        weeklyRoutes    += stats["routes"]
        print(f"done ({stats['runtime_s']:.1f}s)  "
              f"Van {stats['van_routes']}×{stats['van_miles']} mi  "
              f"ST {stats['st_routes']}×{stats['st_miles']} mi")

    print("\nWeekly Summary")
    print("-" * 60)
    print(f"  Total routes: {weeklyRoutes} "
          f"(Van: {sum(len(vanByDay[d]) for d in DAYS)}  "
          f"ST: {sum(len(stByDay[d]) for d in DAYS)})")
    print(f"  Total miles:  {weeklyMiles:,}")
    print(f"  Annual miles: {weeklyMiles * 52:,}")

    # Resource analysis
    combinedByDay = {day: vanByDay[day] + stByDay[day] for day in DAYS}
    analyser      = ResourceAnalyser(combinedByDay)
    analyser.analyse()
    analyser.printReport()

    # Geocode
    print("\nGeocoding ZIPs...")
    allZips = {DEPOT_ZIP}
    for day in DAYS:
        for route in vanByDay[day] + stByDay[day]:
            for stop in route:
                allZips.add(stop["TOZIP"])
    zipCoords = geocodeAllZips(allZips)

    # Folium map
    print("Building interactive map...")
    m       = buildMap(vanByDay, stByDay, zipCoords)
    mapPath = os.path.join(OUTPUT_DIR, "routes_map_mixed_fleet.html")
    m.save(mapPath)
    print(f"  Saved: {mapPath}")

    # CSVs and plots
    print("\nExporting outputs...")
    exportRouteDetails(vanByDay, stByDay)
    exportResourceReport(analyser)
    plotFleetAnalysis(vanByDay, stByDay)


if __name__ == "__main__":
    main()
