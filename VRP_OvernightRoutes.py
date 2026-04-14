"""
VRP_OvernightRoutes.py — Overnight DOT break routing scenario for the NHG dataset.

Standalone script. Seeds from CW + local search, applies overnight pairings greedily,
then reports the final solution with resource requirements.
Requires VRP_DataAnalysis.py to have been run first (data/ must exist).
"""

import os

from vrp_solvers.base import (
    DAYS,
    evaluateRoute,
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def printDayReport(day, routes):
    """Print per-route details for one day; return total miles."""
    print(f"\n{day}")
    print("-" * 60)
    dayTotalMiles = 0

    for i, route in enumerate(routes, start=1):
        r = evaluateRoute(route)
        print(
            f"  Route {i}: {routeIds(route)} | "
            f"miles={r['total_miles']} | cube={r['total_cube']} | "
            f"duty={r['total_duty']}h | feasible={r['overall_feasible']}"
        )
        dayTotalMiles += r["total_miles"]

    print(f"  {day} routes: {len(routes)} | total miles: {dayTotalMiles}")
    return dayTotalMiles


def printOvernightCandidates(routesByDay):
    """Print all improving overnight pairs for each adjacent day pair."""
    for d1, d2 in ADJACENT_DAY_PAIRS:
        candidates = findAllOvernightCandidates(
            routesByDay.get(d1, []), routesByDay.get(d2, [])
        )
        print(f"\nOvernight candidates: {d1} -> {d2}")
        if not candidates:
            print("  No improving overnight pairs found.")
        else:
            for cand in candidates:
                print(
                    f"  {d1} Route {cand['route1_idx'] + 1} + "
                    f"{d2} Route {cand['route2_idx'] + 1} | "
                    f"separate={cand['separate_miles']} mi | "
                    f"overnight={cand['overnight_miles']} mi | "
                    f"savings={cand['savings']} mi"
                )


def printAppliedOvernights(overnightRoutes):
    """Print each applied overnight pairing."""
    print("\nApplied Overnight Routes")
    print("-" * 60)
    if not overnightRoutes:
        print("  No overnight routes applied.")
        return

    for k, ovn in enumerate(overnightRoutes, start=1):
        print(
            f"  Overnight {k}: {ovn['day1']} Route {ovn['route1_idx'] + 1} "
            f"+ {ovn['day2']} Route {ovn['route2_idx'] + 1} | "
            f"day1_orders={routeIds(ovn['route1'])} | "
            f"day2_orders={routeIds(ovn['route2'])} | "
            f"savings={ovn['savings']} mi | "
            f"miles={ovn['results']['total_miles']} | "
            f"feasible={ovn['results']['overall_feasible']}"
        )


def main():
    """Build CW + local search routes, apply overnight pairings, report with resources."""
    orders, _ = loadInputs()

    solver = OvernightSolver(ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True))
    routesByDay, overnightRoutes, usedRoutes = solver.solve(orders)

    weeklyTotalMiles  = 0
    weeklyTotalRoutes = 0

    for day in DAYS:
        weeklyTotalMiles  += printDayReport(day, routesByDay[day])
        weeklyTotalRoutes += len(routesByDay[day])

    print("\nWeekly Summary (before overnight)")
    print("-" * 60)
    print("Total weekly routes:", weeklyTotalRoutes)
    print("Total weekly miles:",  weeklyTotalMiles)

    printOvernightCandidates(routesByDay)
    printAppliedOvernights(overnightRoutes)

    stats = solver.getStats()
    print("\nFinal Summary with Overnight")
    print("-" * 60)
    print("Total routes:",        stats["routes"])
    print("Total miles:",         stats["miles"])
    print("Overnight pairs:  ",   stats["overnight_pairs"])

    analyser = ResourceAnalyser(routesByDay, overnightPairings=overnightRoutes)
    analyser.analyse()
    analyser.printReport()


if __name__ == "__main__":
    main()