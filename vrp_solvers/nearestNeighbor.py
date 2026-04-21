"""
vrp_solvers/nearestNeighbor.py — Nearest-neighbor construction with optional local search.
"""

import time

from vrp_solvers.base import (
    DEPOT_ZIP,
    VAN_CAPACITY,
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
        if dayOrders.empty:
            print("  NearestNeighborSolver: no orders for this day, skipping.")
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
        Greedy construction. Routes are extended stop-by-stop to the nearest
        feasible unvisited stop. When no feasible extension exists, a new route
        is opened from the depot.
        """
        unvisited = list(dayOrders.to_dict("records"))

        # Identify stops that can never be placed on any route — cube alone exceeds
        # van capacity. Without this guard the outer while loop would never empty
        # unvisited and spin forever.
        infeasible = [
            stop for stop in unvisited
            if stop["CUBE"] > VAN_CAPACITY
        ]
        if infeasible:
            ids = [int(s["ORDERID"]) for s in infeasible]
            print(f"  NearestNeighborSolver: {len(infeasible)} order(s) exceed van capacity "
                  f"and cannot be routed — skipping order IDs: {ids}")
            unvisited = [s for s in unvisited if s["CUBE"] <= VAN_CAPACITY]

        routes = []

        while unvisited:
            currentRoute = []
            currentZip   = DEPOT_ZIP
            madeProgress = False

            while True:
                bestStop = None
                bestDist = float("inf")
                bestIdx  = -1

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
                currentZip   = bestStop["TOZIP"]
                madeProgress = True
                unvisited.pop(bestIdx)

            if currentRoute:
                routes.append(currentRoute)
            elif not madeProgress and unvisited:
                # No stop in unvisited can be appended to any new route — all remaining
                # stops are individually infeasible as single-stop routes. Break to avoid
                # an infinite outer loop and report the stranded orders.
                ids = [int(s["ORDERID"]) for s in unvisited]
                print(f"  NearestNeighborSolver: {len(unvisited)} order(s) could not be "
                      f"routed (no feasible single-stop route exists) — order IDs: {ids}")
                break

        return routes

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