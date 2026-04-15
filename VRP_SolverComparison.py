"""
VRP_SolverComparison.py — Compare ten algorithm configurations on the NHG dataset.

Requires VRP_DataAnalysis.py to have been run first (data/ must exist).

Comparison matrix:
    cw_only              CW construction only
    nn_only              Nearest-neighbor construction only
    cw_2opt_oropt        CW + 2-opt + or-opt
    nn_2opt_oropt        NN + 2-opt + or-opt
    tabu_search          CW + LS seed → Tabu Search
    simulated_annealing  CW + LS seed → Simulated Annealing
    alns                 CW + LS seed → ALNS
    cw_overnight         CW + LS + overnight pairings
    alns_overnight       ALNS + overnight pairings
    mixed_fleet          Mixed fleet Van + Straight Truck (CW + LS)
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from vrp_solvers.base import DAYS, loadInputs, evaluateRoute
from vrp_solvers.alns               import ALNSSolver
from vrp_solvers.clarkeWright       import ClarkeWrightSolver
from vrp_solvers.nearestNeighbor    import NearestNeighborSolver
from vrp_solvers.simulatedAnnealing import SimulatedAnnealingSolver
from vrp_solvers.tabuSearch         import TabuSearchSolver
from vrp_solvers.resourceAnalyser   import ResourceAnalyser
from vrp_solvers.mixedFleetSolver   import MixedFleetSolver
from vrp_solvers.overnightSolver    import (
    OvernightSolver,
    applyOvernightImprovements,
)
from vrp_solvers.costModel          import CostModel

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "comparison")

ALGO_LABELS = {
    "cw_only":             "CW Only",
    "nn_only":             "NN Only",
    "cw_2opt_oropt":       "CW + 2-opt/Or-opt",
    "nn_2opt_oropt":       "NN + 2-opt/Or-opt",
    "tabu_search":         "Tabu Search",
    "simulated_annealing": "Simulated Annealing",
    "alns":                "ALNS",
    "cw_overnight":        "CW + Overnight",
    "alns_overnight":      "ALNS + Overnight",
    "mixed_fleet":         "Mixed Fleet (Van+ST)",
}

ALGO_COLORS = {
    "cw_only":             "#AABCC5",
    "nn_only":             "#F4A261",
    "cw_2opt_oropt":       "#2A9D8F",
    "nn_2opt_oropt":       "#06D6A0",
    "tabu_search":         "#457B9D",
    "simulated_annealing": "#E63946",
    "alns":                "#9B5DE5",
    "cw_overnight":        "#F72585",
    "alns_overnight":      "#4CC9F0",
    "mixed_fleet":         "#FFD700",
}

# Day-cab configs that have per-day breakdowns — used in grouped day plots
DAY_CAB_KEYS = [
    "cw_only", "nn_only", "cw_2opt_oropt", "nn_2opt_oropt",
    "tabu_search", "simulated_annealing", "alns",
]

PLOT_STYLE = {
    "figsize":    (13, 6),
    "dpi":        150,
    "titleSize":  13,
    "fontSize":   11,
    "gridAlpha":  0.3,
    "spineColor": "#CCCCCC",
}

plt.rcParams.update({
    "font.size":         PLOT_STYLE["fontSize"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        PLOT_STYLE["gridAlpha"],
    "grid.color":        PLOT_STYLE["spineColor"],
})

def makeDirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def saveFig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def buildSolvers():
    return {
        "cw_only":             ClarkeWrightSolver(useTwoOpt=False, useOrOpt=False),
        "nn_only":             NearestNeighborSolver(useTwoOpt=False, useOrOpt=False),
        "cw_2opt_oropt":       ClarkeWrightSolver(useTwoOpt=True,  useOrOpt=True),
        "nn_2opt_oropt":       NearestNeighborSolver(useTwoOpt=True,  useOrOpt=True),
        "tabu_search":         TabuSearchSolver(),
        "simulated_annealing": SimulatedAnnealingSolver(),
        "alns":                ALNSSolver(),
    }

def buildOvernightSolvers():
    return {
        "cw_overnight":   OvernightSolver(ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)),
        "alns_overnight": OvernightSolver(ALNSSolver()),
    }

def buildMixedFleetSolvers():
    return {"mixed_fleet": MixedFleetSolver()}

def runAll(orders, verbose=True):
    """
    Run all ten configurations.
    Returns (results, convergenceData, alnsWeightHist, resourceReports).
    results[key] contains weekly_miles, weekly_routes, min_drivers,
    min_trucks_peak, runtime_s, weekly_cost, annual_cost.
    """
    cm      = CostModel()
    solvers = buildSolvers()
    allKeys = list(solvers.keys()) + ["cw_overnight", "alns_overnight", "mixed_fleet"]

    results = {
        key: {"days": {}, "weekly_miles": 0, "weekly_routes": 0, "runtime_s": 0.0}
        for key in allKeys
    }
    convergence      = {}
    alnsWeightHist   = {}
    resourceReports  = {}
    solverRoutesByDay = {}

    # Day-cab solvers

    for algoKey, solver in solvers.items():
        if verbose:
            print(f"\n  {ALGO_LABELS[algoKey]}")

        routesByDay = {}

        for day in DAYS:
            dayOrders = orders[orders["DayOfWeek"] == day].copy()
            routes    = solver.solve(dayOrders)
            stats     = solver.getStats()

            routesByDay[day]                  = routes
            results[algoKey]["days"][day]     = {**stats, "day": day}
            results[algoKey]["weekly_miles"]  += stats["miles"]
            results[algoKey]["weekly_routes"] += stats["routes"]
            results[algoKey]["runtime_s"]     += stats["runtime_s"]

            if solver.getConvergence() is not None:
                convergence.setdefault(algoKey, {})[day] = solver.getConvergence()

            if algoKey == "alns":
                alnsWeightHist[day] = solver.getWeightHistory()

            if verbose:
                print(f"    {day}: miles={stats['miles']:6,} | routes={stats['routes']:2d} | "
                      f"feasible={stats['feasible']} | {stats['runtime_s']:.1f}s")

        analyser = ResourceAnalyser(routesByDay)
        analyser.analyse()
        report = analyser.getReport()
        resourceReports[algoKey]            = report
        results[algoKey]["min_drivers"]     = report["min_drivers"]
        results[algoKey]["min_trucks_peak"] = report["min_trucks_peak"]
        solverRoutesByDay[algoKey]          = routesByDay

        # Cost
        bd = cm.weeklyBreakdown(routesByDay)
        results[algoKey]["weekly_cost"] = bd["weekly"]["total"]
        results[algoKey]["annual_cost"] = bd["annual"]["total"]

        if verbose:
            print(f"    Resources: {report['min_drivers']} drivers | "
                  f"{report['min_trucks_peak']} peak trucks | "
                  f"annual cost ${bd['annual']['total']:,.0f}")

    # Overnight configs — apply pairings on top of stored routes

    for overnightKey, baseKey in [("cw_overnight", "cw_2opt_oropt"),
                                   ("alns_overnight", "alns")]:
        if verbose:
            print(f"\n  {ALGO_LABELS[overnightKey]}")

        baseRoutesByDay = solverRoutesByDay.get(baseKey, {})
        if not baseRoutesByDay:
            continue

        overnightRoutes, usedRoutes = applyOvernightImprovements(baseRoutesByDay)

        finalMiles  = 0
        finalRoutes = 0
        allFeasible = True
        for day in DAYS:
            for idx, route in enumerate(baseRoutesByDay.get(day, [])):
                if idx not in usedRoutes[day]:
                    r = evaluateRoute(route)
                    finalMiles  += r["total_miles"]
                    finalRoutes += 1
                    if not r["overall_feasible"]:
                        allFeasible = False
        for ovn in overnightRoutes:
            finalMiles  += ovn["results"]["total_miles"]
            finalRoutes += 1
            if not ovn["results"]["overall_feasible"]:
                allFeasible = False

        analyser = ResourceAnalyser(baseRoutesByDay, overnightPairings=overnightRoutes)
        analyser.analyse()
        report = analyser.getReport()

        bd = cm.weeklyBreakdown(baseRoutesByDay, overnightPairings=overnightRoutes)

        results[overnightKey] = {
            "days":            {},
            "weekly_miles":    finalMiles,
            "weekly_routes":   finalRoutes,
            "runtime_s":       0.0,
            "min_drivers":     report["min_drivers"],
            "min_trucks_peak": report["min_trucks_peak"],
            "weekly_cost":     bd["weekly"]["total"],
            "annual_cost":     bd["annual"]["total"],
        }
        resourceReports[overnightKey] = report

        if verbose:
            print(f"    miles={finalMiles:6,} | routes={finalRoutes:2d} | "
                  f"overnight_pairs={len(overnightRoutes)} | feasible={allFeasible} | "
                  f"drivers={report['min_drivers']} | trucks={report['min_trucks_peak']} | "
                  f"annual cost ${bd['annual']['total']:,.0f}")

    # Mixed fleet

    mixedSolvers = buildMixedFleetSolvers()
    for mKey, mSolver in mixedSolvers.items():
        if verbose:
            print(f"\n  {ALGO_LABELS.get(mKey, mKey)}")

        mRoutesByDay = {}
        vanByDay     = {}
        stByDay      = {}
        totalMiles   = 0
        totalRoutes  = 0
        totalRuntime = 0.0
        allFeasible  = True

        for day in DAYS:
            dayOrders = orders[orders["DayOfWeek"] == day].copy()
            mSolver.solve(dayOrders)
            stats = mSolver.getStats()

            vanByDay[day]      = mSolver.getVanRoutes()
            stByDay[day]       = mSolver.getStRoutes()
            mRoutesByDay[day]  = vanByDay[day] + stByDay[day]
            totalMiles        += stats["miles"]
            totalRoutes       += stats["routes"]
            totalRuntime      += stats["runtime_s"]
            if not stats["feasible"]:
                allFeasible = False

            if verbose:
                print(f"    {day}: miles={stats['miles']:6,} | "
                      f"van={stats['van_routes']} | st={stats['st_routes']} | "
                      f"{stats['runtime_s']:.1f}s")

        analyser = ResourceAnalyser(mRoutesByDay)
        analyser.analyse()
        report = analyser.getReport()

        bd = cm.weeklyBreakdown(routesByDay=None, vanByDay=vanByDay, stByDay=stByDay)

        results[mKey] = {
            "days":            {},
            "weekly_miles":    totalMiles,
            "weekly_routes":   totalRoutes,
            "runtime_s":       round(totalRuntime, 2),
            "min_drivers":     report["min_drivers"],
            "min_trucks_peak": report["min_trucks_peak"],
            "weekly_cost":     bd["weekly"]["total"],
            "annual_cost":     bd["annual"]["total"],
        }
        resourceReports[mKey] = report

        if verbose:
            print(f"    Resources: {report['min_drivers']} drivers | "
                  f"{report['min_trucks_peak']} peak trucks | "
                  f"annual cost ${bd['annual']['total']:,.0f}")

    return results, convergence, alnsWeightHist, resourceReports

def exportCsvs(results):
    """Write comparison_summary.csv and per_day_detail.csv to outputs/comparison/."""
    summaryRows = [
        {
            "algorithm":       key,
            "label":           ALGO_LABELS[key],
            "weekly_miles":    data["weekly_miles"],
            "weekly_routes":   data["weekly_routes"],
            "min_drivers":     data.get("min_drivers", ""),
            "min_trucks_peak": data.get("min_trucks_peak", ""),
            "weekly_cost":     data.get("weekly_cost", ""),
            "annual_cost":     data.get("annual_cost", ""),
            "runtime_s":       round(data["runtime_s"], 2),
        }
        for key, data in results.items()
    ]
    summaryDF = pd.DataFrame(summaryRows).sort_values("weekly_miles")
    summaryDF.to_csv(os.path.join(OUTPUT_DIR, "comparison_summary.csv"), index=False)
    print("  Saved: comparison_summary.csv")

    detailRows = [
        {"algorithm": key, "label": ALGO_LABELS[key], **dayData}
        for key, data in results.items()
        for dayData in data["days"].values()
    ]
    detailDF = pd.DataFrame(detailRows)
    detailDF.to_csv(os.path.join(OUTPUT_DIR, "per_day_detail.csv"), index=False)
    print("  Saved: per_day_detail.csv")

    return summaryDF, detailDF

def plotMilesComparison(summaryDF):
    """Bar chart of total weekly miles per algorithm."""
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    colors  = [ALGO_COLORS[a] for a in summaryDF["algorithm"]]
    bars    = ax.bar(summaryDF["label"], summaryDF["weekly_miles"],
                     color=colors, edgecolor="white", alpha=0.9)

    for bar, v in zip(bars, summaryDF["weekly_miles"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f"{v:,}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Total Weekly Miles")
    ax.set_title("Weekly Total Miles by Algorithm", fontsize=PLOT_STYLE["titleSize"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    saveFig("miles_comparison.png")

def plotCostComparison(summaryDF):
    """Bar chart of annual cost per algorithm."""
    df = summaryDF[summaryDF["annual_cost"] != ""].copy()
    df["annual_cost"] = pd.to_numeric(df["annual_cost"])
    df = df.sort_values("annual_cost")

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    colors  = [ALGO_COLORS[a] for a in df["algorithm"]]
    bars    = ax.bar(df["label"], df["annual_cost"],
                     color=colors, edgecolor="white", alpha=0.9)

    for bar, v in zip(bars, df["annual_cost"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2000,
                f"${v/1e6:.2f}M", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Annual Cost (USD)")
    ax.set_title("Annual Cost by Algorithm", fontsize=PLOT_STYLE["titleSize"])
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x:,.0f}")
    )
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    saveFig("cost_comparison.png")

def plotMilesByDay(detailDF):
    """Grouped bar chart of daily miles — day-cab configs only."""
    algos = DAY_CAB_KEYS
    x     = np.arange(len(DAYS))
    width = 0.12

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, algo in enumerate(algos):
        algoData = detailDF[detailDF["algorithm"] == algo].set_index("day").reindex(DAYS)
        offset   = (i - len(algos) / 2 + 0.5) * width
        ax.bar(x + offset, algoData["miles"].fillna(0).values,
               width, label=ALGO_LABELS[algo],
               color=ALGO_COLORS[algo], edgecolor="white", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(DAYS)
    ax.set_ylabel("Miles")
    ax.set_title("Daily Miles by Algorithm (day-cab configs)", fontsize=PLOT_STYLE["titleSize"])
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    saveFig("miles_by_day.png")

def plotRoutesComparison(detailDF):
    """Grouped bar chart of route count — day-cab configs only."""
    algos = DAY_CAB_KEYS
    x     = np.arange(len(DAYS))
    width = 0.12

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, algo in enumerate(algos):
        algoData = detailDF[detailDF["algorithm"] == algo].set_index("day").reindex(DAYS)
        offset   = (i - len(algos) / 2 + 0.5) * width
        ax.bar(x + offset, algoData["routes"].fillna(0).values,
               width, label=ALGO_LABELS[algo],
               color=ALGO_COLORS[algo], edgecolor="white", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(DAYS)
    ax.set_ylabel("Number of Routes")
    ax.set_title("Route Count by Algorithm per Day (day-cab configs)",
                 fontsize=PLOT_STYLE["titleSize"])
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    saveFig("routes_comparison.png")

def plotRuntimeComparison(summaryDF):
    """Horizontal bar chart of total runtime per algorithm."""
    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = [ALGO_COLORS[a] for a in summaryDF["algorithm"]]
    ax.barh(summaryDF["label"], summaryDF["runtime_s"],
            color=colors, edgecolor="white", alpha=0.9)

    for i, v in enumerate(summaryDF["runtime_s"]):
        ax.text(v + 0.1, i, f"{v:.1f}s", va="center", fontsize=9)

    ax.set_xlabel("Runtime (seconds)")
    ax.set_title("Wall-Clock Runtime by Algorithm", fontsize=PLOT_STYLE["titleSize"])
    plt.tight_layout()
    saveFig("runtime_comparison.png")

def plotConvergence(convergenceData, algoKey, title, filename):
    """Line plot of best-cost convergence over iterations, one line per day."""
    dayColors = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D", "#9B5DE5"]
    data      = convergenceData.get(algoKey, {})

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    for (day, curve), color in zip(data.items(), dayColors):
        ax.plot(curve, label=day, color=color, linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Solution Miles")
    ax.set_title(title, fontsize=PLOT_STYLE["titleSize"])
    ax.legend(title="Day")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    saveFig(filename)

def plotAlnsWeights(alnsWeightHist):
    """Destroy and repair operator weight evolution averaged across days."""
    destroyNames = list(next(iter(alnsWeightHist.values()))["destroy"].keys())
    repairNames  = list(next(iter(alnsWeightHist.values()))["repair"].keys())

    def avgCurve(opType, opName):
        curves = [alnsWeightHist[day][opType][opName] for day in alnsWeightHist]
        minLen = min(len(c) for c in curves)
        return np.mean([c[:minLen] for c in curves], axis=0)

    dColors = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D"]
    rColors = ["#9B5DE5", "#F72585"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, color in zip(destroyNames, dColors):
        axes[0].plot(avgCurve("destroy", name), label=name, color=color, linewidth=1.8)
    axes[0].set_title("Destroy Operator Weights", fontsize=PLOT_STYLE["titleSize"])
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Weight")
    axes[0].legend()

    for name, color in zip(repairNames, rColors):
        axes[1].plot(avgCurve("repair", name), label=name, color=color, linewidth=1.8)
    axes[1].set_title("Repair Operator Weights", fontsize=PLOT_STYLE["titleSize"])
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Weight")
    axes[1].legend()

    fig.suptitle("ALNS Operator Weight Evolution (avg across days)",
                 fontsize=PLOT_STYLE["titleSize"], y=1.01)
    plt.tight_layout()
    saveFig("operator_weights_alns.png")

def main():
    print("VRP Solver Comparison")
    print("=" * 60)
    makeDirs()

    orders, _ = loadInputs()

    print("\nRunning all algorithms...")
    results, convergenceData, alnsWeightHist, resourceReports = runAll(orders, verbose=True)

    print("\nExporting CSVs...")
    summaryDF, detailDF = exportCsvs(results)

    print("\nGenerating plots...")
    plotMilesComparison(summaryDF)
    plotCostComparison(summaryDF)
    plotMilesByDay(detailDF)
    plotRoutesComparison(detailDF)
    plotRuntimeComparison(summaryDF)
    plotConvergence(convergenceData, "tabu_search",
                    "Tabu Search Convergence", "convergence_tabu.png")
    plotConvergence(convergenceData, "simulated_annealing",
                    "Simulated Annealing Convergence", "convergence_sa.png")
    plotConvergence(convergenceData, "alns",
                    "ALNS Convergence", "convergence_alns.png")
    plotAlnsWeights(alnsWeightHist)

    print("\nWeekly Summary")
    print("-" * 60)
    for _, row in summaryDF.iterrows():
        cost = f"${row['annual_cost']:>12,.0f}" if row["annual_cost"] != "" else "           N/A"
        print(f"  {row['label']:24s} | miles={row['weekly_miles']:6,} | "
              f"routes={row['weekly_routes']:3d} | "
              f"drivers={str(row['min_drivers']):>3} | trucks={str(row['min_trucks_peak']):>3} | "
              f"annual={cost} | {row['runtime_s']:.1f}s")

    print(f"\nAll outputs written to outputs/comparison/")

if __name__ == "__main__":
    main()