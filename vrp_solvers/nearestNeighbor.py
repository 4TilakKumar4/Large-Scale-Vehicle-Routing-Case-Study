"""
vrp_solvers/nearestNeighbor.py — Nearest-neighbor construction with optional local search.
"""

import time

from vrp_solvers.base import (
    DEPOT_ZIP,
    applyLocalSearch,
    consolidateRoutes,
    evaluateRoute,
    getDistance,
)


class NearestNeighborSolver:
    """
    Greedy nearest-neighbor construction heuristic.
    At each step the closest unvisited stop that keeps the route feasible is appended.
    A new route is opened when no feasible extension exists.
    Optionally applies 2-opt + or-opt local search after construction.
    Call solve() per day; retrieve results with getStats() and getConvergence().
    """

    def __init__(self, useTwoOpt=True, useOrOpt=True):
        self.useTwoOpt = useTwoOpt
        self.useOrOpt  = useOrOpt
        self._stats    = None

    def solve(self, dayOrders):
        """Build routes for one day and return the final route list."""
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
        Greedy construction. Routes are extended stop-by-stop to the nearest
        feasible unvisited stop. When no feasible extension exists, a new route
        is opened from the depot.
        """
        unvisited = list(dayOrders.to_dict("records"))
        routes    = []

        while unvisited:
            currentRoute = []
            currentZip   = DEPOT_ZIP

            while True:
                bestStop  = None
                bestDist  = float("inf")
                bestIdx   = -1

                for idx, stop in enumerate(unvisited):
                    candidate = currentRoute + [stop]
                    if evaluateRoute(candidate)["overall_feasible"]:
                        d = getDistance(currentZip, stop["TOZIP"])
                        if d < bestDist:
                            bestDist = d
                            bestStop = stop
                            bestIdx  = idx

                if bestStop is None:
                    break

                currentRoute.append(bestStop)
                currentZip = bestStop["TOZIP"]
                unvisited.pop(bestIdx)

            if currentRoute:
                routes.append(currentRoute)

        return routes

    def _collectStats(self, routes, elapsed):
        totalMiles  = sum(evaluateRoute(r)["total_miles"] for r in routes)
        allFeasible = all(evaluateRoute(r)["overall_feasible"] for r in routes)
        return {
            "miles":     totalMiles,
            "routes":    len(routes),
            "feasible":  allFeasible,
            "runtime_s": round(elapsed, 2),
        }
