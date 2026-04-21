"""
VRP_MixedFleet.py — Mixed-fleet routing scenario for the NHG dataset.

Sub-problem 2: Vans (3,200 ft³) and Straight Trucks (1,400 ft³).
ST-required orders must travel on a Straight Truck; all others may use either.
Requires VRP_DataAnalysis.py to have been run first (data/ must exist).

Usage:
    python VRP_MixedFleet.py              # solver + interactive Folium map
    python VRP_MixedFleet.py --no-map    # solver only (faster)

Outputs:
    outputs/mixed_fleet/route_details.csv            — per-stop timing (vehicle type column)
    outputs/mixed_fleet/resource_summary.csv         — headline resource metrics
    outputs/mixed_fleet/driver_chains.csv            — driver chain assignments
    outputs/mixed_fleet/fleet_analysis.png           — daily miles + capacity utilisation
    outputs/mixed_fleet/route_map.png                — static 2×5 grid route map
    outputs/mixed_fleet/routes_map_mixed_fleet.html  — interactive Folium map (unless --no-map)
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import colorsys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    toClock,
)
from vrp_solvers.mixedFleetSolver import MixedFleetSolver, _getMiles

COUNTRY_CODE = "US"
MAP_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "outputs", "mixed_fleet",
                            "routes_map_mixed_fleet.html")
from vrp_solvers.resourceAnalyser import ResourceAnalyser
from vrp_solvers.costModel        import CostModel

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "mixed_fleet")

VAN_DAY_COLORS = {"Mon":"#e53935","Tue":"#fb8c00","Wed":"#1e88e5","Thu":"#8e24aa","Fri":"#43a047"}
ST_DAY_COLORS  = {"Mon":"#ff8a80","Tue":"#ffd180","Wed":"#82b1ff","Thu":"#ea80fc","Fri":"#b9f6ca"}
ROUTE_PALETTE  = ["#E63946","#F4A261","#2A9D8F","#457B9D","#9B5DE5","#F72585","#4CC9F0","#06D6A0"]

DAY_COLORS = {
    "Mon": {"van": "#e53935", "st": "#ff8a80"},
    "Tue": {"van": "#fb8c00", "st": "#ffd180"},
    "Wed": {"van": "#1e88e5", "st": "#82b1ff"},
    "Thu": {"van": "#8e24aa", "st": "#ea80fc"},
    "Fri": {"van": "#43a047", "st": "#b9f6ca"},
}

# Console report

def printDayReport(day, vanRoutes, stRoutes):
    """Print per-route details for one day; return (total_miles, total_routes)."""
    print(f"\n{day}")
    print("-" * 60)
    dayMiles  = 0
    dayRoutes = 0

    for i, route in enumerate(vanRoutes, start=1):
        r    = evaluateMixedRoute(route, "van")
        oids = routeIds(route)
        print(
            f"  Van R{i}: {oids} | orders={len(route)} | "
            f"miles={r['total_miles']} | cube={r['total_cube']}/{VAN_CAPACITY} | "
            f"drive={r['total_drive']}h | duty={r['total_duty']}h | "
            f"feasible={r['overall_feasible']}"
        )
        dayMiles  += r["total_miles"]
        dayRoutes += 1

    for i, route in enumerate(stRoutes, start=1):
        r    = evaluateMixedRoute(route, "st")
        oids = routeIds(route)
        print(
            f"  ST  R{i}: {oids} | orders={len(route)} | "
            f"miles={r['total_miles']} | cube={r['total_cube']}/{ST_CAPACITY} | "
            f"drive={r['total_drive']}h | duty={r['total_duty']}h | "
            f"feasible={r['overall_feasible']}"
        )
        dayMiles  += r["total_miles"]
        dayRoutes += 1

    print(f"  {day} — Van: {len(vanRoutes)} routes | ST: {len(stRoutes)} routes | "
          f"Total miles: {dayMiles}")
    return dayMiles, dayRoutes

def verifySolution(vanByDay, stByDay):
    """Print constraint audit table — mirrors the case paper SolutionCheck tool."""
    print()
    print("  " + "─" * 78)
    print("  SOLUTION VERIFICATION")
    print("  " + "─" * 78)
    print(f"  {'Route':<14} {'Veh':<4} {'Stops':>5} {'Miles':>7} "
          f"{'Drive':>8} {'Duty':>7} {'Cube':>6} {'Cap':>5}  Status")
    print("  " + "-" * 78)

    allServed  = []
    violations = []

    for day in DAYS:
        for ri, route in enumerate(vanByDay.get(day, []), start=1):
            r    = evaluateMixedRoute(route, "van")
            cube = int(r["total_cube"])
            tag  = f"{day} Van R{ri}"
            iss  = _routeIssues(r, cube, "van")
            if iss:
                violations.append((tag, iss))
            _printRow(tag, "Van", len(route), r, cube, VAN_CAPACITY, iss)
            allServed.extend(routeIds(route))

        for ri, route in enumerate(stByDay.get(day, []), start=1):
            r    = evaluateMixedRoute(route, "st")
            cube = int(r["total_cube"])
            tag  = f"{day} ST  R{ri}"
            iss  = _routeIssues(r, cube, "st")
            if iss:
                violations.append((tag, iss))
            _printRow(tag, "ST", len(route), r, cube, ST_CAPACITY, iss)
            allServed.extend(routeIds(route))

    print("  " + "-" * 78)

    orderOk = len(allServed) == 261 and len(set(allServed)) == 261
    allPass = orderOk and not violations

    checks = [
        ("261 orders served, none duplicated",
         orderOk),
        ("Van capacity ≤ 3,200 ft³ per route",
         all(evaluateMixedRoute(r, "van")["capacity_feasible"]
             for d in DAYS for r in vanByDay.get(d, []))),
        ("ST capacity  ≤ 1,400 ft³ per route",
         all(evaluateMixedRoute(r, "st")["capacity_feasible"]
             for d in DAYS for r in stByDay.get(d, []))),
        ("DOT drive ≤ 11 h per shift",
         not any("DRIVE" in i for _, il in violations for i in il)),
        ("DOT duty  ≤ 14 h per shift",
         not any("DUTY"  in i for _, il in violations for i in il)),
        ("All routes feasible",
         not violations),
    ]

    print()
    print(f"  {'Constraint':<48} Status")
    print(f"  {'-' * 48} {'-' * 8}")
    for desc, ok in checks:
        print(f"  {desc:<48} {'✓  PASS' if ok else '✗  FAIL'}")
    print(f"  {'─' * 58}")
    print(f"  {'✓  ALL CONSTRAINTS SATISFIED' if allPass else '✗  VIOLATIONS FOUND'}")
    if violations:
        for tag, iss in violations:
            print(f"     {tag}: {', '.join(iss)}")
    print(f"  {'─' * 58}")

def _routeIssues(res, cube, vehicleType):
    cap    = VAN_CAPACITY if vehicleType == "van" else ST_CAPACITY
    issues = []
    if not res["overall_feasible"]:  issues.append("INFEASIBLE")
    if cube > cap:                   issues.append(f"CAP {cube}>{cap}")
    if res["total_drive"] > 11.01:   issues.append(f"DRIVE {res['total_drive']:.2f}h")
    if res["total_duty"]  > 14.01:   issues.append(f"DUTY {res['total_duty']:.2f}h")
    return issues

def _printRow(tag, veh, stops, res, cube, cap, issues):
    status = "✓ OK" if not issues else "✗ FAIL"
    print(f"  {tag:<14} {veh:<4} {stops:>5} {res['total_miles']:>7} "
          f"{res['total_drive']:>8.2f}h {res['total_duty']:>7.2f}h "
          f"{cube:>6} {cap:>5}  {status}")

# Exports

def exportRouteDetails(vanByDay, stByDay):
    """Write per-stop timing CSV with vehicle_type column."""
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
    """Write driver chains and truck counts to CSV."""
    summaryDF, chainsDF = analyser.toDataFrame()
    summaryPath = os.path.join(OUTPUT_DIR, "resource_summary.csv")
    chainsPath  = os.path.join(OUTPUT_DIR, "driver_chains.csv")
    summaryDF.to_csv(summaryPath, index=False)
    chainsDF.to_csv(chainsPath,   index=False)
    print(f"  Saved: {summaryPath}")
    print(f"  Saved: {chainsPath}")

# Plots

def plotFleetAnalysis(vanByDay, stByDay):
    """Stacked daily miles and average capacity utilisation per fleet."""
    vanMi   = [sum(_getMiles(r, "van") for r in vanByDay.get(d, [])) for d in DAYS]
    stMi    = [sum(_getMiles(r, "st")  for r in stByDay.get(d, []))  for d in DAYS]
    vanUtil = [
        sum(float(s["CUBE"]) for r in vanByDay.get(d, []) for s in r)
        / max(1, len(vanByDay.get(d, [])) * VAN_CAPACITY) * 100
        for d in DAYS
    ]
    stUtil  = [
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
    ax.set_xticks(x)
    ax.set_xticklabels(DAYS)
    ax.set_ylabel("Miles", color="white")
    ax.set_title("Daily Miles — Van vs Straight Truck", color="white", fontweight="bold")
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
    ax2.set_xticks(x)
    ax2.set_xticklabels(DAYS)
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

def plotRouteMap(vanByDay, stByDay):
    """2×5 static grid — Van (top row) and ST (bottom row) routes per day."""
    locsPath = os.path.join(DATA_DIR, "locations_clean.csv")
    if not os.path.exists(locsPath):
        print("  Skipping route map — locations_clean.csv not found.")
        return

    locs = pd.read_csv(locsPath)
    zipToCoord = {
        int(row["ZIP"]): (float(row["lat"]), float(row["lon"]))
        for _, row in locs.iterrows()
    }

    if DEPOT_ZIP not in zipToCoord:
        print("  Skipping route map — depot ZIP not in locations.")
        return

    depotCoord  = zipToCoord[DEPOT_ZIP]
    weeklyTotal = (sum(_getMiles(r, "van") for d in DAYS for r in vanByDay.get(d, []))
                   + sum(_getMiles(r, "st")  for d in DAYS for r in stByDay.get(d, [])))

    def mkShade(hexC, idx, total):
        r2, g2, b2 = (int(hexC[1:3], 16) / 255,
                      int(hexC[3:5], 16) / 255,
                      int(hexC[5:7], 16) / 255)
        h, s, v    = colorsys.rgb_to_hsv(r2, g2, b2)
        v2         = 0.4 + idx / max(1, total - 1) * 0.45
        rr, gg, bb = colorsys.hsv_to_rgb(h, s, v2)
        return f"#{int(rr * 255):02x}{int(gg * 255):02x}{int(bb * 255):02x}"

    fig, axes = plt.subplots(2, 5, figsize=(26, 11))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        f"Mixed Fleet Route Map  ·  Van (top), ST (bottom)  ·  "
        f"Weekly: {weeklyTotal:,.0f} mi  ·  Annual: {weeklyTotal * 52:,.0f} mi",
        color="white", fontsize=11, fontweight="bold", y=1.01,
    )

    for col, day in enumerate(DAYS):
        for row, (routeList, vehicleType, baseHex, label) in enumerate([
            (vanByDay.get(day, []), "van", DAY_COLORS[day]["van"], "Van"),
            (stByDay.get(day,  []), "st",  DAY_COLORS[day]["st"],  "ST"),
        ]):
            ax    = axes[row][col]
            ax.set_facecolor("#0d1b2a")
            n     = len(routeList)
            dayMi = sum(_getMiles(r, vehicleType) for r in routeList)

            for ri, route in enumerate(routeList):
                c    = mkShade(baseHex, ri, n)
                path = [DEPOT_ZIP] + [int(s["TOZIP"]) for s in route] + [DEPOT_ZIP]
                lons = [zipToCoord[z][1] for z in path if z in zipToCoord]
                lats = [zipToCoord[z][0] for z in path if z in zipToCoord]
                ax.plot(lons, lats, "-", color=c, lw=2.0, zorder=3, alpha=0.9)

                for seq, stop in enumerate(route):
                    coord = zipToCoord.get(int(stop["TOZIP"]))
                    if not coord:
                        continue
                    marker = "o" if vehicleType == "van" else "s"
                    ax.scatter(coord[1], coord[0], s=32, color=c,
                               marker=marker, zorder=4)
                    ax.text(coord[1], coord[0], str(seq + 1),
                            ha="center", va="center", fontsize=4.5,
                            color="white", fontweight="bold", zorder=5)
                ax.plot([], [], color=c, lw=2, label=f"R{ri + 1}")

            ax.scatter(depotCoord[1], depotCoord[0],
                       s=220, color="#f5a623", marker="*", zorder=6)
            ax.text(depotCoord[1], depotCoord[0] + 0.06, "DC",
                    ha="center", fontsize=7, color="#f5a623",
                    fontweight="bold", zorder=7)

            ttl = f"{day} {label}  ·  {n} routes  ·  {dayMi:.0f} mi"
            ax.set_title(ttl, color="white", fontsize=8.5, fontweight="bold", pad=4)
            ax.tick_params(colors="#445", labelsize=5.5)
            for sp in ax.spines.values():
                sp.set_color("#223")
            if col == 0:
                ax.set_ylabel("Van Routes" if row == 0 else "ST Routes",
                              color="#aaa", fontsize=8)
            ax.legend(loc="lower right", fontsize=5, framealpha=0.6,
                      facecolor="#0d1b2a", labelcolor="white", edgecolor="#334")

    plt.tight_layout()
    outPath = os.path.join(OUTPUT_DIR, "route_map.png")
    plt.savefig(outPath, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  Saved: {outPath}")

def geocodeAllZips(allZips):
    """Geocode ZIPs to (lat, lon); MDS fallback for misses."""
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


def buildMap(vanByDay, stByDay, zipCoords):
    """Build a Folium map with Van and ST FeatureGroups per day."""
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
        vanColor = VAN_DAY_COLORS[day]
        stColor  = ST_DAY_COLORS[day]

        vanGroup = folium.FeatureGroup(name=f"{day} — Van", show=True)
        for rIdx, route in enumerate(vanByDay.get(day, [])):
            routeColor = ROUTE_PALETTE[rIdx % len(ROUTE_PALETTE)]
            r          = evaluateMixedRoute(route, "van")
            label      = f"{day} · Van R{rIdx + 1}"
            waypoints  = ([depotCoord]
                          + [zipCoords[s["TOZIP"]] for s in route if s["TOZIP"] in zipCoords]
                          + [depotCoord])
            folium.PolyLine(
                locations=waypoints, color=routeColor, weight=3, opacity=0.9,
                tooltip=(f"{label} | {r['total_miles']} mi | "
                         f"{len(route)} orders | cube={r['total_cube']}/{VAN_CAPACITY} ft³ | "
                         f"drive={r['total_drive']}h | duty={r['total_duty']}h"),
            ).add_to(vanGroup)
            plugins.AntPath(locations=waypoints, color=routeColor, weight=3,
                            opacity=0.5, delay=1200, dash_array=[10, 40]).add_to(vanGroup)
            for seq, stop in enumerate(route, start=1):
                z = stop["TOZIP"]
                if z not in zipCoords: continue
                coord     = zipCoords[z]
                oid       = int(stop["ORDERID"])
                popupHtml = (f'<div style="font-family:monospace;font-size:13px;min-width:190px;">'
                             f"<b>{label}</b><br>Stop #{seq}<br>"
                             f"Order ID: {oid}<br>ZIP: {int(z)}<br>"
                             f"Cube: {int(stop['CUBE'])} ft³<br>Vehicle: Van<br>"
                             f"Feasible: {r['overall_feasible']}</div>")
                folium.CircleMarker(location=coord, radius=7, color="white", weight=2,
                    fill=True, fill_color=routeColor, fill_opacity=0.9,
                    tooltip=f"Van Stop {seq} · Order {oid}",
                    popup=folium.Popup(popupHtml, max_width=230)).add_to(vanGroup)
                folium.Marker(location=coord, icon=folium.DivIcon(
                    html=(f'<div style="font-size:9px;font-weight:bold;color:white;'
                          f'text-align:center;line-height:14px;">{seq}</div>'),
                    icon_size=(14, 14), icon_anchor=(7, 7))).add_to(vanGroup)
        vanGroup.add_to(m)

        stGroup = folium.FeatureGroup(name=f"{day} — ST", show=True)
        for rIdx, route in enumerate(stByDay.get(day, [])):
            r     = evaluateMixedRoute(route, "st")
            label = f"{day} · ST R{rIdx + 1}"
            waypoints = ([depotCoord]
                         + [zipCoords[s["TOZIP"]] for s in route if s["TOZIP"] in zipCoords]
                         + [depotCoord])
            folium.PolyLine(
                locations=waypoints, color=stColor, weight=3, opacity=0.9,
                dash_array="6 4",
                tooltip=(f"{label} | {r['total_miles']} mi | "
                         f"{len(route)} orders | cube={r['total_cube']}/{ST_CAPACITY} ft³ | "
                         f"drive={r['total_drive']}h | duty={r['total_duty']}h"),
            ).add_to(stGroup)
            for seq, stop in enumerate(route, start=1):
                z = stop["TOZIP"]
                if z not in zipCoords: continue
                coord     = zipCoords[z]
                oid       = int(stop["ORDERID"])
                popupHtml = (f'<div style="font-family:monospace;font-size:13px;min-width:190px;">'
                             f"<b>{label}</b><br>Stop #{seq}<br>"
                             f"Order ID: {oid}<br>ZIP: {int(z)}<br>"
                             f"Cube: {int(stop['CUBE'])} ft³<br>Vehicle: ST<br>"
                             f"Feasible: {r['overall_feasible']}</div>")
                folium.CircleMarker(location=coord, radius=7, color="white", weight=2,
                    fill=True, fill_color=stColor, fill_opacity=0.9,
                    tooltip=f"ST Stop {seq} · Order {oid}",
                    popup=folium.Popup(popupHtml, max_width=230)).add_to(stGroup)
                folium.Marker(location=coord, icon=folium.DivIcon(
                    html=(f'<div style="font-size:9px;font-weight:bold;color:#0f1117;'
                          f'text-align:center;line-height:14px;">{seq}</div>'),
                    icon_size=(14, 14), icon_anchor=(7, 7))).add_to(stGroup)
        stGroup.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main():
    """Run mixed fleet solver, report results, export outputs; optionally build Folium map."""
    parser = argparse.ArgumentParser(description="NHG Mixed Fleet — Van + Straight Truck")
    parser.add_argument("--no-map", action="store_true",
                        help="Skip geocoding and map generation")
    args = parser.parse_args()

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
    print(f"  Total routes:  {weeklyRoutes} "
          f"(Van: {sum(len(vanByDay[d]) for d in DAYS)}  "
          f"ST: {sum(len(stByDay[d]) for d in DAYS)})")
    print(f"  Total miles:   {weeklyMiles:,}")
    print(f"  Annual miles:  {weeklyMiles * 52:,}")

    for day in DAYS:
        printDayReport(day, vanByDay[day], stByDay[day])

    verifySolution(vanByDay, stByDay)

    # Resource analysis — combined route set
    combinedByDay = {day: vanByDay[day] + stByDay[day] for day in DAYS}
    analyser      = ResourceAnalyser(combinedByDay)
    analyser.analyse()
    analyser.printReport()

    costModel = CostModel()
    breakdown = costModel.weeklyBreakdown(
        routesByDay=None, vanByDay=vanByDay, stByDay=stByDay
    )
    costModel.printSummary(breakdown, label="Mixed Fleet (Van + ST)")

    print("\nExporting outputs...")
    exportRouteDetails(vanByDay, stByDay)
    exportResourceReport(analyser)
    plotFleetAnalysis(vanByDay, stByDay)
    plotRouteMap(vanByDay, stByDay)

    if not args.no_map:
        print("\nGeocoding ZIPs...")
        allZips = {DEPOT_ZIP}
        for day in DAYS:
            for route in vanByDay[day] + stByDay[day]:
                for stop in route:
                    allZips.add(stop["TOZIP"])
        zipCoords = geocodeAllZips(allZips)
        print("Building interactive map...")
        buildMap(vanByDay, stByDay, zipCoords).save(MAP_FILE)
        print(f"  Saved: {MAP_FILE}")
    else:
        print("\n(Map skipped — run without --no-map to generate)")


if __name__ == "__main__":
    main()