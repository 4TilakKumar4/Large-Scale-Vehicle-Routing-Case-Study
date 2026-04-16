"""
VRP_RelaxedSchedule.py — Q3: Relaxed delivery day scheduling for the NHG dataset.

Treats store delivery-day assignments as decision variables rather than fixed
inputs. Three assignment methods are compared, each paired with two routers,
giving six configurations total. All routes are single-day (return to depot
each night) — this is a pure day-assignment optimisation, not overnight routing.

Configurations
--------------
  fixed_cw        Original day assignments  + CW + LS         (Q1 baseline)
  sweep_cw        Angular sweep assignment  + CW + LS
  cpsat_cw        CP-SAT assignment         + CW + LS
  fixed_ort       Original day assignments  + OR-Tools
  sweep_ort       Angular sweep assignment  + OR-Tools
  cpsat_ort       CP-SAT assignment         + OR-Tools         (best expected)

Requires VRP_DataAnalysis.py to have been run first (data/ must exist).

Usage:
    python VRP_RelaxedSchedule.py                    # all six configs
    python VRP_RelaxedSchedule.py --no-map           # skip map
    python VRP_RelaxedSchedule.py --ort-time 30      # OR-Tools time limit/day
    python VRP_RelaxedSchedule.py --cp-time  30      # CP-SAT time limit
    python VRP_RelaxedSchedule.py --configs fixed_cw sweep_cw cpsat_cw
                                                     # run a subset

Outputs:
    outputs/relaxed_schedule/results_summary.csv     — all six configs
    outputs/relaxed_schedule/assignment_comparison.csv
    outputs/relaxed_schedule/route_details_<config>.csv
    outputs/relaxed_schedule/comparison_chart.png
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from vrp_solvers.base import (
    DATA_DIR,
    DAYS,
    DEPOT_ZIP,
    detailedRouteTrace,
    evaluateRoute,
    loadInputs,
    routeIds,
)
from vrp_solvers.angularSweepAssigner import AngularSweepAssigner, loadAngularSweepInputs
from vrp_solvers.cpSatAssigner        import CpSatAssigner
from vrp_solvers.clarkeWright         import ClarkeWrightSolver
from vrp_solvers.orToolsSolver        import ORToolsSolver
from vrp_solvers.resourceAnalyser     import ResourceAnalyser
from vrp_solvers.costModel            import CostModel

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "relaxed_schedule")

ALL_CONFIGS = ["fixed_cw", "sweep_cw", "cpsat_cw", "fixed_ort", "sweep_ort", "cpsat_ort"]

CONFIG_LABELS = {
    "fixed_cw":   "Fixed + CW (baseline)",
    "sweep_cw":   "Sweep + CW",
    "cpsat_cw":   "CP-SAT + CW",
    "fixed_ort":  "Fixed + OR-Tools",
    "sweep_ort":  "Sweep + OR-Tools",
    "cpsat_ort":  "CP-SAT + OR-Tools",
}

CONFIG_COLORS = {
    "fixed_cw":   "#AABCC5",
    "sweep_cw":   "#2A9D8F",
    "cpsat_cw":   "#457B9D",
    "fixed_ort":  "#F4A261",
    "sweep_ort":  "#E63946",
    "cpsat_ort":  "#9B5DE5",
}

PLOT_STYLE = {
    "figsize":   (13, 6),
    "dpi":       150,
    "titleSize": 13,
    "fontSize":  11,
    "gridAlpha": 0.3,
    "spineColor":"#CCCCCC",
}

plt.rcParams.update({
    "font.size":         PLOT_STYLE["fontSize"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        PLOT_STYLE["gridAlpha"],
    "grid.color":        PLOT_STYLE["spineColor"],
})


# ── assignment helpers ──────────────────────────────────────────────────────

def getAssignedOrders(method, originalOrders, locs, cpTimeSec):
    """Return revised orders DataFrame for the given assignment method."""
    if method == "fixed":
        return originalOrders.copy()

    if method == "sweep":
        sweeper = AngularSweepAssigner(originalOrders, locs)
        return sweeper.assign()

    if method == "cpsat":
        assigner = CpSatAssigner(originalOrders, locs, timeLimitSec=cpTimeSec)
        return assigner.assign()

    raise ValueError(f"Unknown assignment method: {method}")


# ── per-config runner ───────────────────────────────────────────────────────

def runConfig(configKey, orders, locs, cpTimeSec, ortTimeSec, verbose):
    """
    Run one configuration end-to-end.
    Returns (routesByDay, stats dict).
    """
    assignMethod, routerKey = configKey.split("_")   # e.g. "cpsat", "ort"

    # Step 1: day assignment
    t0           = time.time()
    revisedOrders = getAssignedOrders(assignMethod, orders, locs, cpTimeSec)
    assignTime   = time.time() - t0

    # Step 2: build solver
    if routerKey == "cw":
        solver = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)
    else:
        solver = ORToolsSolver(timeLimitSec=ortTimeSec)

    # Step 3: solve each day
    routesByDay  = {}
    totalMiles   = 0
    totalRoutes  = 0
    routeTime    = 0.0

    for day in DAYS:
        dayOrders       = revisedOrders[revisedOrders["DayOfWeek"] == day].copy()
        routes          = solver.solve(dayOrders)
        stats           = solver.getStats()
        routesByDay[day] = routes
        totalMiles      += stats["miles"]
        totalRoutes     += stats["routes"]
        routeTime       += stats["runtime_s"]
        allFeasible      = stats["feasible"]

        if verbose:
            print(f"    {day}: {stats['routes']:2d} routes | "
                  f"{stats['miles']:5,} mi | feasible={stats['feasible']} | "
                  f"{stats['runtime_s']:.1f}s")

    # Step 4: resource and cost analysis
    analyser = ResourceAnalyser(routesByDay)
    analyser.analyse()
    report = analyser.getReport()

    cm        = CostModel()
    breakdown = cm.weeklyBreakdown(routesByDay)

    # Step 5: feasibility check across all routes
    allFeasible = all(
        evaluateRoute(r)["overall_feasible"]
        for routes in routesByDay.values()
        for r in routes
    )

    stats = {
        "config":          configKey,
        "label":           CONFIG_LABELS[configKey],
        "weekly_miles":    totalMiles,
        "annual_miles":    totalMiles * 52,
        "weekly_routes":   totalRoutes,
        "min_drivers":     report["min_drivers"],
        "peak_trucks":     report["min_trucks_peak"],
        "weekly_cost":     breakdown["weekly"]["total"],
        "annual_cost":     breakdown["annual"]["total"],
        "assign_time_s":   round(assignTime,  2),
        "routing_time_s":  round(routeTime,   2),
        "total_time_s":    round(assignTime + routeTime, 2),
        "all_feasible":    allFeasible,
    }

    return routesByDay, stats


# ── reporting ───────────────────────────────────────────────────────────────

def printConfigReport(configKey, routes, stats):
    print(f"\n{CONFIG_LABELS[configKey]}")
    print("-" * 60)
    for day in DAYS:
        dayMiles = sum(evaluateRoute(r)["total_miles"] for r in routes.get(day, []))
        print(f"  {day}: {len(routes.get(day, [])):2d} routes | {dayMiles:5,} mi")
    print(f"  Weekly: {stats['weekly_miles']:,} mi | {stats['weekly_routes']} routes | "
          f"drivers={stats['min_drivers']} | trucks={stats['peak_trucks']}")
    print(f"  Annual cost: ${stats['annual_cost']:,.0f} | "
          f"feasible={stats['all_feasible']} | "
          f"time={stats['total_time_s']:.1f}s")


def printComparisonTable(allStats, baselineKey="fixed_cw"):
    """Full comparison table with savings vs baseline."""
    baselineMiles = allStats[baselineKey]["weekly_miles"]
    baselineCost  = allStats[baselineKey]["annual_cost"]

    print("\nQ3 Relaxed Schedule — Full Comparison")
    print("-" * 100)
    print(f"  {'Config':<22} {'W.Miles':>8} {'Saved mi':>9} {'Saved%':>7} "
          f"{'Annual $':>12} {'Saved $':>10} {'Routes':>7} "
          f"{'Drivers':>8} {'Feasible':>9}")
    print(f"  {'-'*22} {'-'*8} {'-'*9} {'-'*7} "
          f"{'-'*12} {'-'*10} {'-'*7} {'-'*8} {'-'*9}")

    for configKey in ALL_CONFIGS:
        if configKey not in allStats:
            continue
        s         = allStats[configKey]
        savedMi   = baselineMiles - s["weekly_miles"]
        savedPct  = savedMi / baselineMiles * 100 if baselineMiles else 0
        savedCost = baselineCost  - s["annual_cost"]
        marker    = " ←" if configKey == baselineKey else ""
        print(
            f"  {s['label']:<22} {s['weekly_miles']:>8,} {savedMi:>+9,} "
            f"{savedPct:>+6.1f}% {s['annual_cost']:>12,.0f} "
            f"{savedCost:>+10,.0f} {s['weekly_routes']:>7} "
            f"{s['min_drivers']:>8} {str(s['all_feasible']):>9}{marker}"
        )

    print(f"\n  Baseline: {CONFIG_LABELS[baselineKey]}")


def printAssignmentImpact(allStats):
    """Isolate the contribution of assignment method vs router."""
    print("\nAssignment Method Impact (CW router held constant)")
    print("-" * 60)
    baseMi = allStats.get("fixed_cw", {}).get("weekly_miles", 0)
    for k in ["fixed_cw", "sweep_cw", "cpsat_cw"]:
        if k not in allStats:
            continue
        mi   = allStats[k]["weekly_miles"]
        diff = baseMi - mi
        pct  = diff / baseMi * 100 if baseMi else 0
        print(f"  {CONFIG_LABELS[k]:<25} {mi:>8,} mi  {diff:>+7,} ({pct:>+.1f}%)")

    print("\nRouter Impact (CP-SAT assignment held constant)")
    print("-" * 60)
    baseMi = allStats.get("cpsat_cw", {}).get("weekly_miles", 0)
    for k in ["cpsat_cw", "cpsat_ort"]:
        if k not in allStats:
            continue
        mi   = allStats[k]["weekly_miles"]
        diff = baseMi - mi
        pct  = diff / baseMi * 100 if baseMi else 0
        print(f"  {CONFIG_LABELS[k]:<25} {mi:>8,} mi  {diff:>+7,} ({pct:>+.1f}%)")


# ── export ──────────────────────────────────────────────────────────────────

def exportRouteDetails(configKey, routesByDay):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    locsPath = os.path.join(DATA_DIR, "locations_clean.csv")
    locs     = pd.read_csv(locsPath) if os.path.exists(locsPath) else None

    allRows = []
    for day in DAYS:
        for routeNum, route in enumerate(routesByDay.get(day, []), start=1):
            allRows.extend(detailedRouteTrace(route, day, routeNum, locs))

    df = pd.DataFrame(allRows, columns=[
        "day", "route_number", "stop_sequence", "order_id",
        "location", "arrival_time", "departure_time", "delivery_volume_cuft",
    ])
    path = os.path.join(OUTPUT_DIR, f"route_details_{configKey}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def exportResultsSummary(allStats):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = [s for s in allStats.values()]
    df   = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "results_summary.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def exportAssignmentComparison(originalOrders, locs, cpTimeSec):
    """Write before/after cube distribution for all three assignment methods."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target = originalOrders["CUBE"].sum() / len(DAYS)
    rows   = []

    for method, label in [("fixed",  "Fixed"),
                           ("sweep",  "Angular Sweep"),
                           ("cpsat",  "CP-SAT")]:
        if method == "cpsat":
            revised = CpSatAssigner(
                originalOrders, locs, timeLimitSec=cpTimeSec
            ).assign()
        elif method == "sweep":
            revised = AngularSweepAssigner(originalOrders, locs).assign()
        else:
            revised = originalOrders.copy()

        for day in DAYS:
            cube = revised[revised["DayOfWeek"] == day]["CUBE"].sum()
            rows.append({
                "method":       label,
                "day":          day,
                "cube":         int(cube),
                "target":       int(target),
                "pct_of_target": round(cube / target * 100, 1),
            })

    df   = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "assignment_comparison.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


def buildComparisonChart(allStats):
    """Bar chart: weekly miles for all six configurations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    configs = [k for k in ALL_CONFIGS if k in allStats]
    miles   = [allStats[k]["weekly_miles"] for k in configs]
    labels  = [CONFIG_LABELS[k]            for k in configs]
    colors  = [CONFIG_COLORS[k]            for k in configs]

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])

    bars = ax.bar(labels, miles, color=colors, width=0.55, edgecolor="white")

    # Annotate each bar with its value
    for bar, mi in zip(bars, miles):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{mi:,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Baseline reference line
    baselineMi = allStats.get("fixed_cw", {}).get("weekly_miles", 0)
    if baselineMi:
        ax.axhline(baselineMi, color="#333333", linewidth=1.2,
                   linestyle="--", alpha=0.6, label=f"Baseline {baselineMi:,} mi")
        ax.legend(fontsize=9)

    ax.set_ylabel("Weekly Miles")
    ax.set_title("Q3 Relaxed Schedule — Weekly Miles by Configuration",
                 fontsize=PLOT_STYLE["titleSize"], fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "comparison_chart.png")
    plt.savefig(path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NHG Q3 Relaxed Schedule — six assignment × router configurations"
    )
    parser.add_argument("--ort-time", type=int, default=30,
                        help="OR-Tools time limit per day in seconds (default: 30)")
    parser.add_argument("--cp-time",  type=int, default=30,
                        help="CP-SAT assignment time limit in seconds (default: 30)")
    parser.add_argument("--configs",  nargs="+", default=ALL_CONFIGS,
                        choices=ALL_CONFIGS,
                        help="Subset of configs to run (default: all six)")
    parser.add_argument("--no-map",   action="store_true",
                        help="Skip map generation")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    originalOrders, _ = loadInputs()
    _, locs           = loadAngularSweepInputs()

    allStats      = {}
    allRoutesByDay = {}

    for configKey in args.configs:
        print(f"\nRunning {CONFIG_LABELS[configKey]}...")
        print("-" * 60)

        routesByDay, stats = runConfig(
            configKey, originalOrders, locs,
            cpTimeSec=args.cp_time,
            ortTimeSec=args.ort_time,
            verbose=True,
        )
        allStats[configKey]       = stats
        allRoutesByDay[configKey] = routesByDay

        printConfigReport(configKey, routesByDay, stats)

    # ── summary tables ──────────────────────────────────────────────────
    if len(allStats) > 1:
        printComparisonTable(allStats)
        printAssignmentImpact(allStats)

    # ── exports ─────────────────────────────────────────────────────────
    print("\nExporting outputs...")
    for configKey, routesByDay in allRoutesByDay.items():
        exportRouteDetails(configKey, routesByDay)

    exportResultsSummary(allStats)

    print("  Building assignment balance comparison...")
    exportAssignmentComparison(originalOrders, locs, args.cp_time)

    if len(allStats) > 1:
        print("  Building comparison chart...")
        buildComparisonChart(allStats)


if __name__ == "__main__":
    main()
