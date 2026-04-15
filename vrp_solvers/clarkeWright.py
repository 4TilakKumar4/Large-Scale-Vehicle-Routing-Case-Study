"""
vrp_solvers/clarkeWright.py — Clarke-Wright savings construction with optional local search.
"""

import time
from itertools import combinations

from vrp_solvers.base import (
    DEPOT_ZIP,
    applyLocalSearch,
    consolidateRoutes,
    evaluateRoute,
    getDistance,
)


class ClarkeWrightSolver:
    """
    Clarke-Wright parallel savings algorithm.
    Optionally applies 2-opt + or-opt local search after construction.
    Call solve() per day; retrieve results with getStats() and getConvergence().
    """

    def __init__(self, useTwoOpt=True, useOrOpt=True):
        self.useTwoOpt = useTwoOpt
        self.useOrOpt  = useOrOpt
        self._stats    = None

    def solve(self, dayOrders):
        """Build routes for one day and return the final route list."""
        if dayOrders.empty:
            print("  ClarkeWrightSolver: no orders for this day, skipping.")
            self._stats = {"miles": 0, "routes": 0, "feasible": True, "runtime_s": 0.0}
            return []

        t0 = time.time()

        routes = self._build(dayOrders)
        routes = consolidateRoutes(routes)

        if self.useTwoOpt or self.useOrOpt:
            routes = applyLocalSearch(routes)

        self._stats = self._collectStats(routes, time.time() - t0)
        return routes

    def getStats(self):
        """Return miles, routes, runtime, and feasibility from the last solve() call."""
        return self._stats

    def getConvergence(self):
        """Construction heuristics have no iterative convergence curve."""
        return None

    def _build(self, dayOrders):
        """
        Parallel savings construction. Each stop begins as its own route;
        pairs merge in descending savings order when the result stays feasible.
        Only valid endpoint adjacencies are attempted (tail-to-head, with reversal).
        """
        orderRecords = {int(row["ORDERID"]): row for _, row in dayOrders.iterrows()}

        routes  = {oid: [rec] for oid, rec in orderRecords.items()}
        routeOf = {oid: oid   for oid in orderRecords}
        headOf  = {oid: oid   for oid in orderRecords}
        tailOf  = {oid: oid   for oid in orderRecords}

        for s, oidI, oidJ in self._computeSavings(dayOrders):
            if s <= 0:
                break

            if oidI not in routeOf or oidJ not in routeOf:
                continue

            ridI = routeOf[oidI]
            ridJ = routeOf[oidJ]

            if ridI == ridJ:
                continue

            routeI = routes[ridI]
            routeJ = routes[ridJ]

            candidates = []
            if tailOf[ridI] == oidI and headOf[ridJ] == oidJ:
                candidates.append((routeI + routeJ,       ridI, ridJ))
            if tailOf[ridJ] == oidJ and headOf[ridI] == oidI:
                candidates.append((routeJ + routeI,       ridJ, ridI))
            if tailOf[ridI] == oidI and tailOf[ridJ] == oidJ:
                candidates.append((routeI + routeJ[::-1], ridI, ridJ))
            if headOf[ridI] == oidI and headOf[ridJ] == oidJ:
                candidates.append((routeI[::-1] + routeJ, ridI, ridJ))

            for mergedRoute, keepRid, dropRid in candidates:
                if evaluateRoute(mergedRoute)["overall_feasible"]:
                    routes[keepRid] = mergedRoute
                    del routes[dropRid]

                    headOf[keepRid] = int(mergedRoute[0]["ORDERID"])
                    tailOf[keepRid] = int(mergedRoute[-1]["ORDERID"])

                    for stop in mergedRoute:
                        routeOf[int(stop["ORDERID"])] = keepRid

                    del headOf[dropRid]
                    del tailOf[dropRid]
                    break

        return list(routes.values())

    def _computeSavings(self, ordersDF):
        """s(i,j) = d(depot,i) + d(depot,j) - d(i,j), sorted descending."""
        orderList = ordersDF.to_dict("records")
        savings   = []

        for a, b in combinations(orderList, 2):
            zipA = a["TOZIP"]
            zipB = b["TOZIP"]
            s = (
                getDistance(DEPOT_ZIP, zipA)
                + getDistance(DEPOT_ZIP, zipB)
                - getDistance(zipA, zipB)
            )
            savings.append((s, int(a["ORDERID"]), int(b["ORDERID"])))

        savings.sort(key=lambda x: x[0], reverse=True)
        return savings

    def _collectStats(self, routes, elapsed):
        if not routes:
            return {"miles": 0, "routes": 0, "feasible": True, "runtime_s": round(elapsed, 2)}

        totalMiles  = sum(evaluateRoute(r)["total_miles"] for r in routes)
        allFeasible = all(evaluateRoute(r)["overall_feasible"] for r in routes)
        return {
            "miles":     totalMiles,
            "routes":    len(routes),
            "feasible":  allFeasible,
            "runtime_s": round(elapsed, 2),
        }