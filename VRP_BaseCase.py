"""
VRP_BaseCase.py — Runs Clarke-Wright + 2-opt + or-opt on the NHG dataset.

Requires VRP_DataAnalysis.py to have been run first (data/ must exist).
Outputs:
  outputs/base_case/route_details.csv   — per-stop timing in Table 3 format
  outputs/base_case/resource_report.csv — driver chains and truck counts
"""

import os

import pandas as pd

from vrp_solvers.base import (
    DATA_DIR,
    DAYS,
    detailedRouteTrace,
    evaluateRoute,
    loadInputs,
    routeIds,
)
from vrp_solvers.clarkeWright import ClarkeWrightSolver
from vrp_solvers.resourceAnalyser import ResourceAnalyser

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "base_case")


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
    """
    Write per-stop timing CSV in Table 3 format for all routes across all days.
    Output: outputs/base_case/route_details.csv
    """
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

    outPath = os.path.join(OUTPUT_DIR, "route_details.csv")
    detailDF.to_csv(outPath, index=False)
    print(f"  Saved: {outPath}")


def exportResourceReport(analyser):
    """
    Write driver chains and truck counts to CSV.
    Output: outputs/base_case/resource_report.csv (summary) and
            outputs/base_case/driver_chains.csv
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summaryDF, chainsDF = analyser.toDataFrame()

    summaryPath = os.path.join(OUTPUT_DIR, "resource_summary.csv")
    chainsPath  = os.path.join(OUTPUT_DIR, "driver_chains.csv")

    summaryDF.to_csv(summaryPath, index=False)
    chainsDF.to_csv(chainsPath,   index=False)

    print(f"  Saved: {summaryPath}")
    print(f"  Saved: {chainsPath}")


def main():
    """Build CW + local search routes for each day, report results, and export outputs."""
    orders, _ = loadInputs()
    solver    = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)

    routesByDay = {}
    for day in DAYS:
        print(f"Building routes for {day}...")
        dayOrders        = orders[orders["DayOfWeek"] == day].copy()
        routesByDay[day] = solver.solve(dayOrders)

    weeklyTotalMiles  = 0
    weeklyTotalRoutes = 0
    weeklyTotalOrders = 0

    for day in DAYS:
        dayMiles, dayOrders = printDayReport(day, routesByDay[day])
        weeklyTotalMiles  += dayMiles
        weeklyTotalRoutes += len(routesByDay[day])
        weeklyTotalOrders += dayOrders

    print("\nWeekly Summary")
    print("-" * 60)
    print("Total routes:",           weeklyTotalRoutes)
    print("Total orders fulfilled:", weeklyTotalOrders)
    print("Total miles:",            weeklyTotalMiles)

    analyser = ResourceAnalyser(routesByDay)
    analyser.analyse()
    analyser.printReport()

    print("\nExporting outputs...")
    exportRouteDetails(routesByDay)
    exportResourceReport(analyser)


if __name__ == "__main__":
    main()