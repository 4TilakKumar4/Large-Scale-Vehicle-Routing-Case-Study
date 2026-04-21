"""
VRP_CostAnalysis.py — Cost estimation and sensitivity analysis across all routing scenarios.

Reads pre-built routes from each scenario script and applies CostModel to produce:
  - Full weekly and annual cost breakdown per scenario
  - Algorithm cost comparison (from VRP_SolverComparison results)
  - Sensitivity analysis plots across key cost parameters

Requires all scenario scripts to have been run first so route data is available.
All default rates: Northeast US dedicated contract carrier, 2024-25 benchmarks.

Outputs (outputs/cost_analysis/):
  cost_summary.csv        — weekly + annual cost by scenario
  cost_breakdown.csv      — per-component weekly cost by scenario
  sensitivity_summary.csv — cost at each parameter value in the sensitivity sweep
  cost_comparison.png     — bar chart: annual cost by algorithm
  cost_breakdown.png      — stacked bar: cost components by scenario
  sensitivity_*.png       — one sensitivity plot per parameter
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from vrp_solvers.base import (
    DAYS,
    evaluateRoute,
    evaluateMixedRoute,
    loadInputs,
)
from vrp_solvers.clarkeWright       import ClarkeWrightSolver
from vrp_solvers.alns               import ALNSSolver
from vrp_solvers.mixedFleetSolver   import MixedFleetSolver, ALNSMixedFleetSolver
from vrp_solvers.overnightSolver    import OvernightSolver, applyOvernightImprovements
from vrp_solvers.base               import loadZipCoords
from vrp_solvers.costModel          import CostModel, SENSITIVITY_RANGES
from vrp_solvers.relaxedScheduleSolver import SweepRelaxedSolver, ALNSRelaxedSolver

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "cost_analysis")

PLOT_STYLE = {
    "figsize":    (14, 7),
    "dpi":        150,
    "titleSize":  13,
}

SCENARIO_COLORS = {
    "Base Case (CW+LS)":          "#2A9D8F",
    "Overnight (CW+LS)":          "#457B9D",
    "Mixed Fleet (Van+ST)":        "#F4A261",
    "ALNS":                        "#9B5DE5",
    "ALNS + Overnight":            "#F72585",
}

COMPONENT_COLORS = {
    "mileage":   "#4CC9F0",
    "labour":    "#F72585",
    "benefits":  "#9B5DE5",
    "equipment": "#F4A261",
    "insurance": "#2A9D8F",
    "per_diem":  "#E63946",
}

COMPONENT_LABELS = {
    "mileage":   "Mileage (fuel, R&M, tyres)",
    "labour":    "Driver wages",
    "benefits":  "Driver benefits (30%)",
    "equipment": "Equipment (cab + trailer)",
    "insurance": "Insurance",
    "per_diem":  "Per diem (overnight)",
}

WEEKS_PER_YEAR = 52

# Scenario builders — re-run solvers to get route objects

def buildBaseCase(orders):
    """CW + local search, no overnight."""
    solver      = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)
    routesByDay = {}
    for day in DAYS:
        dayOrders        = orders[orders["DayOfWeek"] == day].copy()
        routesByDay[day] = solver.solve(dayOrders)
    return routesByDay, None, None, None

def buildOvernightCase(orders):
    """CW + local search + overnight pairings."""
    solver = OvernightSolver(ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True))
    routesByDay, overnightRoutes, _ = solver.solve(orders)
    return routesByDay, overnightRoutes, None, None

def buildMixedFleet(orders):
    """Mixed fleet Van + ST, no overnight."""
    solver   = MixedFleetSolver()
    vanByDay = {}
    stByDay  = {}
    for day in DAYS:
        dayOrders = orders[orders["DayOfWeek"] == day].copy()
        solver.solve(dayOrders)
        vanByDay[day] = solver.getVanRoutes()
        stByDay[day]  = solver.getStRoutes()
    return None, None, vanByDay, stByDay

def buildALNSMixedFleet(orders):
    """ALNS mixed fleet Van + ST."""
    solver   = ALNSMixedFleetSolver()
    vanByDay = {}
    stByDay  = {}
    for day in DAYS:
        dayOrders = orders[orders["DayOfWeek"] == day].copy()
        solver.solve(dayOrders)
        vanByDay[day] = solver.getVanRoutes()
        stByDay[day]  = solver.getStRoutes()
    return None, None, vanByDay, stByDay


def buildRelaxedSweep(orders):
    """Angular sweep seed + greedy local search."""
    solver = SweepRelaxedSolver()
    solver.solve(orders)
    finalOrders = solver.getOrders()
    from vrp_solvers.base import solveOneDay as _sod
    routesByDay = {day: _sod(finalOrders[finalOrders["DayOfWeek"] == day].copy())
                   for day in DAYS}
    return routesByDay, None, None, None


def buildRelaxedALNS(orders):
    """ALNS inter-day + greedy local search."""
    solver = ALNSRelaxedSolver()
    solver.solve(orders)
    finalOrders = solver.getOrders()
    from vrp_solvers.base import solveOneDay as _sod
    routesByDay = {day: _sod(finalOrders[finalOrders["DayOfWeek"] == day].copy())
                   for day in DAYS}
    return routesByDay, None, None, None


def buildALNS(orders):
    """ALNS metaheuristic, no overnight."""
    solver      = ALNSSolver()
    routesByDay = {}
    for day in DAYS:
        dayOrders        = orders[orders["DayOfWeek"] == day].copy()
        routesByDay[day] = solver.solve(dayOrders)
    return routesByDay, None, None, None

def buildALNSOvernight(orders):
    """ALNS metaheuristic + overnight pairings."""
    solver = OvernightSolver(ALNSSolver())
    routesByDay, overnightRoutes, _ = solver.solve(orders)
    return routesByDay, overnightRoutes, None, None

SCENARIOS = {
    "Base Case (CW+LS)":   buildBaseCase,
    "Overnight (CW+LS)":   buildOvernightCase,
    "Mixed Fleet (Van+ST)": buildMixedFleet,
    "ALNS":                 buildALNS,
    "ALNS + Overnight":     buildALNSOvernight,
    "ALNS Mixed Fleet":     buildALNSMixedFleet,
    "Relaxed (Sweep+LS)":   buildRelaxedSweep,
    "Relaxed (ALNS+LS)":    buildRelaxedALNS,
}

# Helpers

def computeWeeklyMiles(routesByDay, overnightRoutes=None,
                       vanByDay=None, stByDay=None):
    """Compute total weekly miles for a scenario."""
    used = {}
    for p in (overnightRoutes or []):
        used.setdefault(p["day1"], set()).add(p["route1_idx"])
        used.setdefault(p["day2"], set()).add(p["route2_idx"])

    total = 0
    if vanByDay is not None:
        for day in DAYS:
            for r in vanByDay.get(day, []):
                total += evaluateMixedRoute(r, "van")["total_miles"]
            for r in stByDay.get(day, []):
                total += evaluateMixedRoute(r, "st")["total_miles"]
    else:
        for day in DAYS:
            for idx, r in enumerate(routesByDay.get(day, [])):
                if idx in used.get(day, set()):
                    continue
                total += evaluateRoute(r)["total_miles"]
        for p in (overnightRoutes or []):
            total += p["results"]["total_miles"]
    return total

# Plots

def plotCostComparison(summaryDF):
    """Bar chart of annual total cost by scenario."""
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    colors  = [SCENARIO_COLORS.get(s, "#AABCC5") for s in summaryDF["scenario"]]
    bars    = ax.bar(summaryDF["scenario"], summaryDF["annual_total"],
                     color=colors, edgecolor="white", alpha=0.9)

    for bar, v in zip(bars, summaryDF["annual_total"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5000,
                f"${v:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Annual Cost (USD)")
    ax.set_title("Annual Total Cost by Scenario", fontsize=PLOT_STYLE["titleSize"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
    plt.xticks(rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "cost_comparison.png")
    plt.savefig(path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plotCostBreakdown(breakdownDF):
    """Stacked horizontal bar: annual cost components by scenario."""
    components = ["mileage", "labour", "benefits", "equipment", "insurance", "per_diem"]
    scenarios  = breakdownDF["scenario"].tolist()

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    left    = np.zeros(len(scenarios))

    for comp in components:
        values = breakdownDF[f"annual_{comp}"].values
        bars   = ax.barh(scenarios, values, left=left,
                         color=COMPONENT_COLORS[comp],
                         label=COMPONENT_LABELS[comp], edgecolor="white", alpha=0.9)
        # Label if wide enough
        for bar, v in zip(bars, values):
            if v > 20000:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"${v/1000:.0f}K", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        left += values

    ax.set_xlabel("Annual Cost (USD)")
    ax.set_title("Annual Cost Breakdown by Scenario", fontsize=PLOT_STYLE["titleSize"])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "cost_breakdown.png")
    plt.savefig(path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plotSensitivity(sensitivityDF, param, lowVal, defaultVal, highVal):
    """Line chart of annual cost vs parameter value for each scenario."""
    paramDF = sensitivityDF[sensitivityDF["parameter"] == param]
    if paramDF.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    for scenario in paramDF["scenario"].unique():
        sDF = paramDF[paramDF["scenario"] == scenario].sort_values("param_value")
        ax.plot(sDF["param_value"], sDF["annual_total"],
                marker="o", linewidth=2,
                color=SCENARIO_COLORS.get(scenario, "#AABCC5"),
                label=scenario)

    ax.axvline(defaultVal, color="#AABCC5", linestyle="--", linewidth=1.5,
               label=f"Default ({defaultVal})")
    ax.set_xlabel(param.replace("_", " ").title())
    ax.set_ylabel("Annual Cost (USD)")
    ax.set_title(f"Sensitivity: {param.replace('_', ' ').title()}",
                 fontsize=PLOT_STYLE["titleSize"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"sensitivity_{param}.png")
    plt.savefig(path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    orders, _ = loadInputs()
    loadZipCoords()

    # 1. Build all scenarios and compute costs

    print("Building scenarios and computing costs...")
    print("-" * 60)

    cm           = CostModel()
    summaryRows  = []
    breakdownRows = []
    builtScenarios = {}

    for scenarioName, builderFn in SCENARIOS.items():
        print(f"  {scenarioName}...", end=" ", flush=True)
        t0 = time.time()

        routesByDay, overnightPairings, vanByDay, stByDay = builderFn(orders)
        builtScenarios[scenarioName] = (routesByDay, overnightPairings, vanByDay, stByDay)

        breakdown = cm.weeklyBreakdown(
            routesByDay=routesByDay,
            overnightPairings=overnightPairings,
            vanByDay=vanByDay,
            stByDay=stByDay,
        )
        w = breakdown["weekly"]
        a = breakdown["annual"]

        weeklyMiles = computeWeeklyMiles(routesByDay, overnightPairings, vanByDay, stByDay)

        summaryRows.append({
            "scenario":       scenarioName,
            "weekly_miles":   weeklyMiles,
            "annual_miles":   weeklyMiles * WEEKS_PER_YEAR,
            "weekly_total":   w["total"],
            "annual_total":   a["total"],
        })

        breakdownRows.append({
            "scenario":         scenarioName,
            "weekly_miles":     weeklyMiles,
            **{f"weekly_{k}": w[k] for k in ["mileage", "labour", "benefits",
                                               "equipment", "insurance", "per_diem", "total"]},
            **{f"annual_{k}":  a[k] for k in ["mileage", "labour", "benefits",
                                               "equipment", "insurance", "per_diem", "total"]},
        })

        cm.printSummary(breakdown, label=scenarioName)
        print(f"done ({time.time() - t0:.1f}s)")

    summaryDF   = pd.DataFrame(summaryRows)
    breakdownDF = pd.DataFrame(breakdownRows)

    summaryPath   = os.path.join(OUTPUT_DIR, "cost_summary.csv")
    breakdownPath = os.path.join(OUTPUT_DIR, "cost_breakdown.csv")
    summaryDF.to_csv(summaryPath,   index=False)
    breakdownDF.to_csv(breakdownPath, index=False)
    print(f"\n  Saved: {summaryPath}")
    print(f"  Saved: {breakdownPath}")

    # 2. Sensitivity analysis — vary each parameter across its range

    print("\nRunning sensitivity analysis...")
    print("-" * 60)

    sensitivityRows = []
    N_STEPS         = 7

    for param, (lowVal, defaultVal, highVal) in SENSITIVITY_RANGES.items():
        print(f"  Sweeping {param} [{lowVal} → {highVal}]...")

        paramValues = np.linspace(lowVal, highVal, N_STEPS)

        for pVal in paramValues:
            trialCM = CostModel(**{param: float(pVal)})

            for scenarioName, (routesByDay, overnightPairings, vanByDay, stByDay) in builtScenarios.items():
                bd = trialCM.weeklyBreakdown(
                    routesByDay=routesByDay,
                    overnightPairings=overnightPairings,
                    vanByDay=vanByDay,
                    stByDay=stByDay,
                )
                sensitivityRows.append({
                    "parameter":   param,
                    "param_value": round(float(pVal), 4),
                    "scenario":    scenarioName,
                    "weekly_total": bd["weekly"]["total"],
                    "annual_total": bd["annual"]["total"],
                })

    sensitivityDF   = pd.DataFrame(sensitivityRows)
    sensitivityPath = os.path.join(OUTPUT_DIR, "sensitivity_summary.csv")
    sensitivityDF.to_csv(sensitivityPath, index=False)
    print(f"  Saved: {sensitivityPath}")

    # 3. Plots

    print("\nGenerating plots...")
    plotCostComparison(summaryDF)
    plotCostBreakdown(breakdownDF)

    for param, (lowVal, defaultVal, highVal) in SENSITIVITY_RANGES.items():
        plotSensitivity(sensitivityDF, param, lowVal, defaultVal, highVal)

    # 4. Final summary table

    print("\nAnnual Cost Summary")
    print("-" * 60)
    print(f"  {'Scenario':<28} {'Weekly Miles':>12}  {'Annual Miles':>12}  {'Annual Cost':>13}")
    print(f"  {'-'*28} {'-'*12}  {'-'*12}  {'-'*13}")
    for _, row in summaryDF.iterrows():
        print(f"  {row['scenario']:<28} {row['weekly_miles']:>12,}  "
              f"{row['annual_miles']:>12,}  ${row['annual_total']:>12,.2f}")

    print(f"\n  All outputs written to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()