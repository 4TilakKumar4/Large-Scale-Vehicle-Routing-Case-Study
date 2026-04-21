"""
VRP_SensitivityAnalysis.py — Operational sensitivity analysis for NHG routing scenarios.

Covers three groups of sensitivities:

  Group A — Demand scaling (modifies orders DataFrame only):
    1. Volume scaling      : ±10%, ±20%, ±30% on all order cubes
    2. Peak week           : Wed + Thu orders scaled +25%
    3. ST mix shift        : proportion of ST-required orders raised to 30% / 50%

  Group B — Fleet and operational parameters (patches vrp_solvers.base constants):
    4. Van capacity        : 2,400 / 3,200 / 3,600 ft³
    5. ST capacity         : 1,000 / 1,400 / 1,800 ft³
    6. Driving speed       : 32 / 40 / 48 mph  (±20%)
    7. Unload rate         : −20% / baseline / +20%

  Group C — DOT and schedule (patches vrp_solvers.base constants):
    8. HOS safety buffer   : 0h / 0.5h / 1h subtracted from MAX_DRIVING and MAX_DUTY
    9. Delivery window     : [08,18] / [06,20] / [00,24]
   10. Overnight toggle    : baseline miles vs overnight-allowed miles (cost delta)

All constant patches are applied via ConstantOverride — a context manager that
restores every original value on exit, even if an exception occurs.
Source files are never modified.

Requires VRP_DataAnalysis.py to have been run first (data/ must exist).
Outputs (outputs/sensitivity/):
  operational_sensitivity.csv — all scenario × parameter × value results
  sensitivity_group_A.png     — demand scaling plots
  sensitivity_group_B.png     — fleet and operational plots
  sensitivity_group_C.png     — DOT and schedule plots
"""

import os
import random
import time
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import vrp_solvers.base as _base

from vrp_solvers.base import DAYS, loadInputs, evaluateRoute, evaluateMixedRoute
from vrp_solvers.clarkeWright       import ClarkeWrightSolver
from vrp_solvers.mixedFleetSolver   import MixedFleetSolver, ALNSMixedFleetSolver
from vrp_solvers.overnightSolver    import OvernightSolver, applyOvernightImprovements
from vrp_solvers.resourceAnalyser   import ResourceAnalyser
from vrp_solvers.costModel          import CostModel

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "sensitivity")

RANDOM_SEED = 42  # for reproducible ST mix shift

PLOT_STYLE = {
    "figsize":   (18, 10),
    "dpi":       150,
    "titleSize": 11,
}

# ConstantOverride — safe context manager for patching base.py module attrs

@contextmanager
def ConstantOverride(**overrides):
    """
    Temporarily patch vrp_solvers.base module-level constants.
    Guarantees restoration of every original value on exit, even if the body
    raises an exception. Source files are never touched.

    Usage:
        with ConstantOverride(DRIVING_SPEED=32, MAX_DRIVING=10):
            routes = solver.solve(dayOrders)
        # All originals restored here.
    """
    originals = {}
    try:
        for name, newVal in overrides.items():
            if not hasattr(_base, name):
                raise AttributeError(
                    f"vrp_solvers.base has no attribute '{name}'. "
                    f"Check spelling against base.py constants."
                )
            originals[name] = getattr(_base, name)
            setattr(_base, name, newVal)
        yield
    finally:
        for name, origVal in originals.items():
            setattr(_base, name, origVal)

# Metric collection helpers

def _solveAndMeasure(orders, label, overnightToo=False):
    """
    Run CW + LS on the given orders DataFrame and return a metrics dict.
    Optionally also runs overnight pairings and reports the combined result.
    """
    solver      = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)
    routesByDay = {}

    for day in DAYS:
        dayOrders        = orders[orders["DayOfWeek"] == day].copy()
        routesByDay[day] = solver.solve(dayOrders)

    # Base case metrics
    weeklyMiles  = sum(
        evaluateRoute(r)["total_miles"]
        for day in DAYS for r in routesByDay[day]
    )
    totalRoutes  = sum(len(routesByDay[d]) for d in DAYS)
    allFeasible  = all(
        evaluateRoute(r)["overall_feasible"]
        for day in DAYS for r in routesByDay[day]
    )

    analyser = ResourceAnalyser(routesByDay)
    analyser.analyse()
    report   = analyser.getReport()

    cm       = CostModel()
    bd       = cm.weeklyBreakdown(routesByDay)

    row = {
        "label":           label,
        "weekly_miles":    weeklyMiles,
        "annual_miles":    weeklyMiles * 52,
        "total_routes":    totalRoutes,
        "min_drivers":     report["min_drivers"],
        "peak_trucks":     report["min_trucks_peak"],
        "feasible_pct":    100.0 if allFeasible else _feasiblePct(routesByDay),
        "weekly_cost":     bd["weekly"]["total"],
        "annual_cost":     bd["annual"]["total"],
        "overnight_pairs": 0,
    }

    if overnightToo:
        overnightRoutes, usedRoutes = applyOvernightImprovements(routesByDay)
        onMiles  = sum(
            evaluateRoute(r)["total_miles"]
            for day in DAYS
            for idx, r in enumerate(routesByDay[day])
            if idx not in usedRoutes.get(day, set())
        ) + sum(o["results"]["total_miles"] for o in overnightRoutes)
        onRoutes = (
            sum(len(routesByDay[d]) - len(usedRoutes.get(d, set())) for d in DAYS)
            + len(overnightRoutes)
        )
        row["overnight_weekly_miles"]  = onMiles
        row["overnight_annual_miles"]  = onMiles * 52
        row["overnight_total_routes"]  = onRoutes
        row["overnight_pairs"]         = len(overnightRoutes)

    return row

def _solveMixedAndMeasure(orders, label):
    """Run MixedFleetSolver and return a metrics dict."""
    solver   = MixedFleetSolver()
    vanByDay = {}
    stByDay  = {}

    for day in DAYS:
        dayOrders = orders[orders["DayOfWeek"] == day].copy()
        solver.solve(dayOrders)
        vanByDay[day] = solver.getVanRoutes()
        stByDay[day]  = solver.getStRoutes()

    weeklyMiles = (
        sum(evaluateMixedRoute(r, "van")["total_miles"]
            for day in DAYS for r in vanByDay[day])
        + sum(evaluateMixedRoute(r, "st")["total_miles"]
              for day in DAYS for r in stByDay[day])
    )
    totalRoutes = sum(
        len(vanByDay[d]) + len(stByDay[d]) for d in DAYS
    )
    combined = {day: vanByDay[day] + stByDay[day] for day in DAYS}
    analyser = ResourceAnalyser(combined)
    analyser.analyse()
    report   = analyser.getReport()

    cm = CostModel()
    bd = cm.weeklyBreakdown(routesByDay=None, vanByDay=vanByDay, stByDay=stByDay)

    return {
        "label":           label,
        "weekly_miles":    weeklyMiles,
        "annual_miles":    weeklyMiles * 52,
        "total_routes":    totalRoutes,
        "van_routes":      sum(len(vanByDay[d]) for d in DAYS),
        "st_routes":       sum(len(stByDay[d])  for d in DAYS),
        "min_drivers":     report["min_drivers"],
        "peak_trucks":     report["min_trucks_peak"],
        "weekly_cost":     bd["weekly"]["total"],
        "annual_cost":     bd["annual"]["total"],
    }

def _solveALNSMixedAndMeasure(orders, label):
    """Run ALNSMixedFleetSolver and return a metrics dict."""
    solver   = ALNSMixedFleetSolver()
    vanByDay = {}
    stByDay  = {}
    totalMiles   = 0
    totalRoutes  = 0
    totalRuntime = 0.0
    allFeasible  = True

    for day in DAYS:
        dayOrders = orders[orders["DayOfWeek"] == day].copy()
        solver.solve(dayOrders)
        stats = solver.getStats()
        vanByDay[day]  = solver.getVanRoutes()
        stByDay[day]   = solver.getStRoutes()
        totalMiles    += stats["miles"]
        totalRoutes   += stats["routes"]
        totalRuntime  += stats["runtime_s"]
        if not stats["feasible"]:
            allFeasible = False

    combined = {day: vanByDay[day] + stByDay[day] for day in DAYS}
    analyser = ResourceAnalyser(combined)
    analyser.analyse()
    report = analyser.getReport()

    cm = CostModel()
    bd = cm.weeklyBreakdown(routesByDay=None, vanByDay=vanByDay, stByDay=stByDay)

    return {
        "label":           label,
        "weekly_miles":    totalMiles,
        "annual_miles":    totalMiles * 52,
        "total_routes":    totalRoutes,
        "van_routes":      sum(len(vanByDay[d]) for d in DAYS),
        "st_routes":       sum(len(stByDay[d])  for d in DAYS),
        "min_drivers":     report["min_drivers"],
        "peak_trucks":     report["min_trucks_peak"],
        "weekly_cost":     bd["weekly"]["total"],
        "annual_cost":     bd["annual"]["total"],
    }


def _feasiblePct(routesByDay):
    total     = 0
    feasible  = 0
    for day in DAYS:
        for r in routesByDay[day]:
            total   += 1
            if evaluateRoute(r)["overall_feasible"]:
                feasible += 1
    return round(100.0 * feasible / max(1, total), 1)

# Group A — Demand scaling

def runGroupA(orders):
    """Volume scaling, peak week, and ST mix shift."""
    print("Group A — Demand scaling")
    rows = []

    # --- A1: Volume scaling ---
    print("  A1: Volume scaling...")
    baseline = _solveAndMeasure(orders, "Baseline")
    baseline["sensitivity"] = "A1_volume_scaling"
    baseline["param_value"] = 1.0
    rows.append(baseline)

    for scale in [1.10, 1.20, 1.30]:
        scaled = orders.copy()
        scaled["CUBE"] = (scaled["CUBE"] * scale).round().astype(int)
        r = _solveAndMeasure(scaled, f"+{int((scale-1)*100)}% volume")
        r["sensitivity"] = "A1_volume_scaling"
        r["param_value"] = scale
        rows.append(r)

    # --- A2: Peak week (Wed + Thu +25%) ---
    print("  A2: Peak week (Wed/Thu +25%)...")
    peakDays   = ["Wed", "Thu"]
    peakOrders = orders.copy()
    mask = peakOrders["DayOfWeek"].isin(peakDays)
    peakOrders.loc[mask, "CUBE"] = (peakOrders.loc[mask, "CUBE"] * 1.25).round().astype(int)

    basePeak = _solveAndMeasure(orders, "Baseline (peak week)")
    basePeak["sensitivity"] = "A2_peak_week"
    basePeak["param_value"] = 1.0
    rows.append(basePeak)

    rPeak = _solveAndMeasure(peakOrders, "Wed+Thu +25%")
    rPeak["sensitivity"] = "A2_peak_week"
    rPeak["param_value"] = 1.25
    rows.append(rPeak)

    # --- A3: ST mix shift (CW-based and ALNS-based mixed fleet) ---
    print("  A3: ST mix shift...")
    rng = random.Random(RANDOM_SEED)

    def applySTMix(df, targetFrac):
        df = df.copy()
        flexible = df[df["straight_truck_required"] == "no"].index.tolist()
        currentST = (df["straight_truck_required"] == "yes").sum()
        targetST  = int(len(df) * targetFrac)
        toMark    = max(0, targetST - currentST)
        toMark    = min(toMark, len(flexible))
        chosen    = rng.sample(flexible, toMark)
        df.loc[chosen, "straight_truck_required"] = "yes"
        return df

    currentSTFrac = (orders["straight_truck_required"] == "yes").mean()

    for frac, baseLabel in [(currentSTFrac, f"Current ({currentSTFrac:.0%})"),
                             (0.30, "30% ST"),
                             (0.50, "50% ST")]:
        mixed = applySTMix(orders, frac)

        # CW-based mixed fleet
        r = _solveMixedAndMeasure(mixed, f"CW {baseLabel}")
        r["sensitivity"] = "A3_st_mix_shift"
        r["param_value"] = frac
        r["solver"]      = "CW_Mixed"
        rows.append(r)

        # ALNS-based mixed fleet
        ra = _solveALNSMixedAndMeasure(mixed, f"ALNS {baseLabel}")
        ra["sensitivity"] = "A3_st_mix_shift"
        ra["param_value"] = frac
        ra["solver"]      = "ALNS_Mixed"
        rows.append(ra)

    return rows

# Group B — Fleet and operational parameters

def runGroupB(orders):
    """Van capacity, ST capacity, driving speed, unload rate."""
    print("Group B — Fleet and operational parameters")
    rows = []

    # --- B4: Van capacity ---
    print("  B4: Van capacity...")
    for cap in [2400, 3200, 3600]:
        with ConstantOverride(VAN_CAPACITY=cap):
            r = _solveAndMeasure(orders, f"Van cap {cap} ft³")
        r["sensitivity"] = "B4_van_capacity"
        r["param_value"] = cap
        rows.append(r)

    # --- B5: ST capacity ---
    print("  B5: ST capacity...")
    for cap in [1000, 1400, 1800]:
        with ConstantOverride(ST_CAPACITY=cap):
            r = _solveMixedAndMeasure(orders, f"ST cap {cap} ft³")
        r["sensitivity"] = "B5_st_capacity"
        r["param_value"] = cap
        rows.append(r)

    # --- B6: Driving speed ---
    print("  B6: Driving speed...")
    for speed in [32, 40, 48]:
        with ConstantOverride(DRIVING_SPEED=speed):
            r = _solveAndMeasure(orders, f"{speed} mph")
        r["sensitivity"] = "B6_driving_speed"
        r["param_value"] = speed
        rows.append(r)

    # --- B7: Unload rate ---
    print("  B7: Unload rate (Van + ST)...")
    for mult, label in [(0.80, "−20% unload"), (1.00, "Baseline"), (1.20, "+20% unload")]:
        with ConstantOverride(
            UNLOAD_RATE    = round(0.030 * mult, 4),
            ST_UNLOAD_RATE = round(0.043 * mult, 4),
        ):
            r = _solveAndMeasure(orders, label)
        r["sensitivity"] = "B7_unload_rate"
        r["param_value"] = mult
        rows.append(r)

    return rows

# Group C — DOT and schedule

def runGroupC(orders):
    """HOS safety buffer, delivery window relaxation, overnight toggle."""
    print("Group C — DOT and schedule")
    rows = []

    # --- C8: HOS safety buffer ---
    print("  C8: HOS safety buffer...")
    for buf, label in [(0.0, "No buffer"), (0.5, "0.5h buffer"), (1.0, "1h buffer")]:
        with ConstantOverride(
            MAX_DRIVING = 11.0 - buf,
            MAX_DUTY    = 14.0 - buf,
        ):
            r = _solveAndMeasure(orders, label)
        r["sensitivity"] = "C8_hos_buffer"
        r["param_value"] = buf
        rows.append(r)

    # --- C9: Delivery window relaxation ---
    print("  C9: Delivery window relaxation...")
    windows = [
        (8,  18, "08:00-18:00 (baseline)"),
        (6,  20, "06:00-20:00"),
        (0,  24, "24-hour warehouse"),
    ]
    for wOpen, wClose, label in windows:
        with ConstantOverride(WINDOW_OPEN=wOpen, WINDOW_CLOSE=wClose):
            r = _solveAndMeasure(orders, label)
        r["sensitivity"] = "C9_delivery_window"
        r["param_value"] = wClose - wOpen   # window width in hours
        rows.append(r)

    # --- C10: Overnight toggle (cost delta) ---
    print("  C10: Overnight toggle...")
    # No overnight
    rNo = _solveAndMeasure(orders, "No overnight")
    rNo["sensitivity"] = "C10_overnight_toggle"
    rNo["param_value"] = 0
    rows.append(rNo)

    # With overnight
    rOn = _solveAndMeasure(orders, "Overnight allowed", overnightToo=True)
    rOn["sensitivity"] = "C10_overnight_toggle"
    rOn["param_value"] = 1
    # Use overnight miles/cost for comparison
    rOn["weekly_miles"] = rOn.get("overnight_weekly_miles", rOn["weekly_miles"])
    rOn["annual_miles"]  = rOn.get("overnight_annual_miles",  rOn["annual_miles"])
    rows.append(rOn)

    return rows

# Plotting

def _axStyle(ax, title, xlabel, ylabel, yFmt="miles"):
    ax.set_title(title, fontsize=PLOT_STYLE["titleSize"])
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    if yFmt == "cost":
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M" if x >= 1e6 else f"${x:,.0f}")
        )
    elif yFmt == "miles":
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )

def _barPlot(ax, labels, values, color, title, xlabel, ylabel, yFmt="miles"):
    bars = ax.bar(labels, values, color=color, edgecolor="white", alpha=0.88)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{v:,.0f}" if yFmt == "miles" else f"${v:,.0f}",
                ha="center", va="bottom", fontsize=7.5)
    _axStyle(ax, title, xlabel, ylabel, yFmt)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)

def plotGroupA(rows):
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 3, figsize=PLOT_STYLE["figsize"])
    fig.suptitle("Group A — Demand Sensitivity", fontsize=12, fontweight="bold")

    # A1 volume scaling — miles and routes
    a1 = df[df["sensitivity"] == "A1_volume_scaling"]
    _barPlot(axes[0][0], a1["label"], a1["annual_miles"], "#2A9D8F",
             "A1: Volume Scaling — Annual Miles", "Volume scale", "Annual miles")
    _barPlot(axes[1][0], a1["label"], a1["total_routes"], "#F4A261",
             "A1: Volume Scaling — Routes", "Volume scale", "Total routes", "miles")

    # A2 peak week
    a2 = df[df["sensitivity"] == "A2_peak_week"]
    _barPlot(axes[0][1], a2["label"], a2["annual_miles"], "#9B5DE5",
             "A2: Peak Week — Annual Miles", "Scenario", "Annual miles")
    _barPlot(axes[1][1], a2["label"], a2["annual_cost"], "#E63946",
             "A2: Peak Week — Annual Cost", "Scenario", "Annual cost ($)", "cost")

    # A3 ST mix shift — van vs ST routes
    a3 = df[df["sensitivity"] == "A3_st_mix_shift"]
    if "van_routes" in a3.columns:
        x   = np.arange(len(a3))
        w   = 0.35
        axes[0][2].bar(x - w/2, a3["van_routes"].values, w,
                       label="Van routes", color="#4CC9F0", edgecolor="white", alpha=0.9)
        axes[0][2].bar(x + w/2, a3["st_routes"].values,  w,
                       label="ST routes",  color="#F4A261", edgecolor="white", alpha=0.9)
        axes[0][2].set_xticks(x)
        axes[0][2].set_xticklabels(a3["label"].values, rotation=15, ha="right", fontsize=8)
        axes[0][2].legend(fontsize=8)
        _axStyle(axes[0][2], "A3: ST Mix Shift — Route Split",
                 "ST fraction", "Routes", "miles")
    _barPlot(axes[1][2], a3["label"], a3["annual_cost"], "#E63946",
             "A3: ST Mix Shift — Annual Cost", "ST fraction", "Annual cost ($)", "cost")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sensitivity_group_A.png")
    plt.savefig(path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plotGroupB(rows):
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 4, figsize=PLOT_STYLE["figsize"])
    fig.suptitle("Group B — Fleet & Operational Sensitivity", fontsize=12, fontweight="bold")

    configs = [
        ("B4_van_capacity",  "B4: Van Capacity",  "Capacity (ft³)", "#2A9D8F"),
        ("B5_st_capacity",   "B5: ST Capacity",   "Capacity (ft³)", "#F4A261"),
        ("B6_driving_speed", "B6: Driving Speed",  "Speed (mph)",    "#9B5DE5"),
        ("B7_unload_rate",   "B7: Unload Rate",   "Rate multiplier","#E63946"),
    ]
    for col, (sens, title, xlabel, color) in enumerate(configs):
        sub = df[df["sensitivity"] == sens]
        _barPlot(axes[0][col], sub["label"], sub["annual_miles"], color,
                 f"{title} — Annual Miles", xlabel, "Annual miles")
        _barPlot(axes[1][col], sub["label"], sub["total_routes"], color,
                 f"{title} — Routes", xlabel, "Total routes", "miles")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sensitivity_group_B.png")
    plt.savefig(path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plotGroupC(rows):
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 3, figsize=PLOT_STYLE["figsize"])
    fig.suptitle("Group C — DOT & Schedule Sensitivity", fontsize=12, fontweight="bold")

    # C8 HOS buffer
    c8 = df[df["sensitivity"] == "C8_hos_buffer"]
    _barPlot(axes[0][0], c8["label"], c8["annual_miles"], "#457B9D",
             "C8: HOS Buffer — Annual Miles", "Buffer (hours)", "Annual miles")
    _barPlot(axes[1][0], c8["label"], c8["total_routes"], "#457B9D",
             "C8: HOS Buffer — Routes", "Buffer (hours)", "Routes", "miles")

    # C9 delivery window
    c9 = df[df["sensitivity"] == "C9_delivery_window"]
    _barPlot(axes[0][1], c9["label"], c9["annual_miles"], "#2A9D8F",
             "C9: Delivery Window — Annual Miles", "Window (hours)", "Annual miles")
    _barPlot(axes[1][1], c9["label"], c9["annual_cost"], "#2A9D8F",
             "C9: Delivery Window — Annual Cost", "Window (hours)", "Annual cost ($)", "cost")

    # C10 overnight toggle
    c10 = df[df["sensitivity"] == "C10_overnight_toggle"]
    _barPlot(axes[0][2], c10["label"], c10["annual_miles"], "#9B5DE5",
             "C10: Overnight Toggle — Annual Miles", "Overnight", "Annual miles")
    # Cost delta bar
    if len(c10) == 2:
        delta_miles = c10.iloc[0]["annual_miles"] - c10.iloc[1]["annual_miles"]
        delta_cost  = (c10.iloc[0]["annual_cost"] - c10.iloc[1]["annual_cost"]
                       if "annual_cost" in c10.columns else 0)
        axes[1][2].bar(["Miles saved", "Cost saved"],
                       [delta_miles, delta_cost],
                       color=["#9B5DE5", "#E63946"], edgecolor="white", alpha=0.9)
        axes[1][2].set_title("C10: Overnight Toggle — Savings",
                              fontsize=PLOT_STYLE["titleSize"])
        axes[1][2].yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        axes[1][2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sensitivity_group_C.png")
    plt.savefig(path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("VRP Operational Sensitivity Analysis")
    print("=" * 60)

    orders, _ = loadInputs()

    allRows = []

    t0 = time.time()
    allRows.extend(runGroupA(orders))
    print(f"  Group A done ({time.time() - t0:.1f}s)\n")

    t1 = time.time()
    allRows.extend(runGroupB(orders))
    print(f"  Group B done ({time.time() - t1:.1f}s)\n")

    t2 = time.time()
    allRows.extend(runGroupC(orders))
    print(f"  Group C done ({time.time() - t2:.1f}s)\n")

    # Export CSV
    df      = pd.DataFrame(allRows)
    csvPath = os.path.join(OUTPUT_DIR, "operational_sensitivity.csv")
    df.to_csv(csvPath, index=False)
    print(f"  Saved: {csvPath}")

    # Plots
    print("\nGenerating plots...")
    plotGroupA(df[df["sensitivity"].str.startswith("A")].to_dict("records"))
    plotGroupB(df[df["sensitivity"].str.startswith("B")].to_dict("records"))
    plotGroupC(df[df["sensitivity"].str.startswith("C")].to_dict("records"))

    # Console summary
    print("\nKey Results")
    print("-" * 60)
    for sens in df["sensitivity"].unique():
        sub = df[df["sensitivity"] == sens]
        print(f"\n  {sens}")
        for _, row in sub.iterrows():
            miles = row.get("annual_miles", 0)
            cost  = row.get("annual_cost", 0)
            routes = row.get("total_routes", 0)
            print(f"    {str(row['label']):<30}  "
                  f"miles={int(miles):>8,}  routes={int(routes):>3}  "
                  f"cost=${cost:>10,.0f}")

    print(f"\n  All outputs written to: {OUTPUT_DIR}")
    print(f"  Total runtime: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()