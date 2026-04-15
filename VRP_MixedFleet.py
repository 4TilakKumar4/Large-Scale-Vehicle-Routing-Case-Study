"""
VRP_MixedFleet.py — Mixed-fleet routing scenario for the NHG dataset.

Sub-problem 2: Vans (3,200 ft³) and Straight Trucks (1,400 ft³).
ST-required orders must travel on a Straight Truck; all others may use either.
Requires VRP_DataAnalysis.py to have been run first (data/ must exist).

Outputs:
  outputs/mixed_fleet/route_details.csv      — per-stop timing (Table 3 format + vehicle type)
  outputs/mixed_fleet/resource_summary.csv   — headline resource metrics
  outputs/mixed_fleet/driver_chains.csv      — driver chain assignments
  outputs/mixed_fleet/fleet_analysis.png     — daily miles + capacity utilisation
  outputs/mixed_fleet/route_map.png          — static 2×5 grid route map
"""

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
from vrp_solvers.resourceAnalyser import ResourceAnalyser

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "mixed_fleet")

DAY_COLORS = {
    "Mon": {"van": "#e53935", "st": "#ff8a80"},
    "Tue": {"van": "#fb8c00", "st": "#ffd180"},
    "Wed": {"van": "#1e88e5", "st": "#82b1ff"},
    "Thu": {"van": "#8e24aa", "st": "#ea80fc"},
    "Fri": {"van": "#43a047", "st": "#b9f6ca"},
}


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run mixed fleet solver for each day, verify, and export all outputs."""
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

    print("\nExporting outputs...")
    exportRouteDetails(vanByDay, stByDay)
    exportResourceReport(analyser)
    plotFleetAnalysis(vanByDay, stByDay)
    plotRouteMap(vanByDay, stByDay)


if __name__ == "__main__":
    main()
