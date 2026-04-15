"""
VRP_OvernightRoutes.py — Overnight DOT break routing scenario for the NHG dataset.

Standalone script. Seeds from CW + local search, applies overnight pairings greedily,
then reports the final solution with resource requirements.
Requires VRP_DataAnalysis.py to have been run first (data/ must exist).
Outputs:
  outputs/overnight/route_details.csv   — per-stop timing for all routes (day-cab + overnight)
  outputs/overnight/resource_summary.csv — headline resource metrics
  outputs/overnight/driver_chains.csv    — driver chain assignments
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
from vrp_solvers.clarkeWright    import ClarkeWrightSolver
from vrp_solvers.overnightSolver import (
    OvernightSolver,
    findAllOvernightCandidates,
    applyOvernightImprovements,
    ADJACENT_DAY_PAIRS,
)
from vrp_solvers.resourceAnalyser import ResourceAnalyser
from vrp_solvers.costModel        import CostModel

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "overnight")


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


def exportRouteDetails(routesByDay, overnightRoutes, usedRoutes):
    """
    Write per-stop timing CSV for all routes — day-cab and overnight.

    Day-cab routes follow the standard Table 3 format.
    Overnight pairings are written as two consecutive route segments sharing the
    same route_number, with an additional 'route_type' column marking them.

    Output: outputs/overnight/route_details.csv
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    locsPath = os.path.join(DATA_DIR, "locations_clean.csv")
    locs     = pd.read_csv(locsPath) if os.path.exists(locsPath) else None

    allRows     = []
    routeNum    = 1

    # overnight pairings — two segments share a route_number, labelled by day
    overnightKeys = set()
    for ovn in overnightRoutes:
        overnightKeys.add((ovn["day1"], ovn["route1_idx"]))
        overnightKeys.add((ovn["day2"], ovn["route2_idx"]))

        day1Rows = detailedRouteTrace(ovn["route1"], ovn["day1"], routeNum, locs)
        day2Rows = detailedRouteTrace(ovn["route2"], ovn["day2"], routeNum, locs)

        for row in day1Rows:
            row["route_type"] = "overnight_day1"
        for row in day2Rows:
            row["route_type"] = "overnight_day2"

        # Remove the closing depot row from day1 and opening depot row from day2
        # so the combined trace reads cleanly as one continuous journey
        if day1Rows:
            day1Rows = day1Rows[:-1]   # drop day1 return-to-depot row
        if day2Rows:
            day2Rows = day2Rows[1:]    # drop day2 dispatch-from-depot row

        allRows.extend(day1Rows)
        allRows.extend(day2Rows)
        routeNum += 1

    # day-cab routes not consumed by overnight pairings
    for day in DAYS:
        for ridx, route in enumerate(routesByDay.get(day, [])):
            if ridx in usedRoutes.get(day, set()):
                continue   # consumed by overnight pairing above
            rows = detailedRouteTrace(route, day, routeNum, locs)
            for row in rows:
                row["route_type"] = "day_cab"
            allRows.extend(rows)
            routeNum += 1

    detailDF = pd.DataFrame(allRows, columns=[
        "day", "route_number", "route_type", "stop_sequence", "order_id",
        "location", "arrival_time", "departure_time", "delivery_volume_cuft",
    ])

    outPath = os.path.join(OUTPUT_DIR, "route_details.csv")
    detailDF.to_csv(outPath, index=False)
    print(f"  Saved: {outPath}")


def exportResourceReport(analyser):
    """
    Write driver chains and truck counts to CSV.
    Output: outputs/overnight/resource_summary.csv
            outputs/overnight/driver_chains.csv
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
    """Build CW + local search routes, apply overnight pairings, report and export outputs."""
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
    print("Total routes:     ", stats["routes"])
    print("Total miles:      ", stats["miles"])
    print("Overnight pairs:  ", stats["overnight_pairs"])

    analyser = ResourceAnalyser(routesByDay, overnightPairings=overnightRoutes)
    analyser.analyse()
    analyser.printReport()

    costModel = CostModel()
    breakdown = costModel.weeklyBreakdown(routesByDay, overnightPairings=overnightRoutes)
    costModel.printSummary(breakdown, label="Overnight Routes (CW + LS)")

    print("\nExporting outputs...")
    exportRouteDetails(routesByDay, overnightRoutes, usedRoutes)
    exportResourceReport(analyser)


if __name__ == "__main__":
    main()