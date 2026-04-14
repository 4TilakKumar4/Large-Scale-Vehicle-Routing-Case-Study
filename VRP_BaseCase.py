"""
VRP_BaseCase.py — Runs Clarke-Wright + 2-opt + or-opt on the NHG dataset.

Requires VRP_DataAnalysis.py to have been run first (data/ must exist).
"""

import os

from vrp_solvers.base import (
    DAYS,
    evaluateRoute,
    loadInputs,
    routeIds,
)
from vrp_solvers.clarkeWright import ClarkeWrightSolver

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


def main():
    """Build CW + local search routes for each day and report results."""
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


if __name__ == "__main__":
    main()
