"""
vrp_solvers/simulatedAnnealing.py — Simulated Annealing metaheuristic seeded from CW + local search.
"""

import math
import random
import time

from vrp_solvers.base import evaluateRoute
from vrp_solvers.clarkeWright import ClarkeWrightSolver


class SimulatedAnnealingSolver:
    """
    Simulated Annealing with geometric cooling.
    Accepts worse solutions with probability exp(-delta/T), enabling escape from local optima.
    Seed is CW + consolidate + 2-opt + or-opt for a fair starting point.
    Call solve() per day; retrieve results with getStats() and getConvergence().
    """

    def __init__(self, maxIter=2000, tempStart=500.0, tempEnd=1.0, randomSeed=42):
        self.maxIter    = maxIter
        self.tempStart  = tempStart
        self.tempEnd    = tempEnd
        self.randomSeed = randomSeed
        self._stats       = None
        self._convergence = None

    def solve(self, dayOrders):
        """Seed from CW, run SA, return best routes found."""
        if dayOrders.empty:
            print("  SimulatedAnnealingSolver: no orders for this day, skipping.")
            self._stats       = {"miles": 0, "routes": 0, "feasible": True, "runtime_s": 0.0}
            self._convergence = []
            return []

        t0 = time.time()
        random.seed(self.randomSeed)

        seed             = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True).solve(dayOrders)
        routes, convergence = self._search(seed)

        self._convergence = convergence
        self._stats       = self._collectStats(routes, time.time() - t0)
        return routes

    def getStats(self):
        return self._stats

    def getConvergence(self):
        return self._convergence

    def _search(self, initRoutes):
        # Nothing to search if the seed is empty
        if not initRoutes:
            return [], []

        currentRoutes = [list(r) for r in initRoutes]
        bestRoutes    = [list(r) for r in initRoutes]
        currentMiles  = self._totalMiles(currentRoutes)
        bestMiles     = currentMiles
        convergence   = [bestMiles]

        # Guard against zero initial miles causing a degenerate temperature
        if currentMiles == 0:
            return bestRoutes, convergence

        coolingRate = (self.tempEnd / self.tempStart) ** (1.0 / self.maxIter)
        T           = self.tempStart

        for _ in range(self.maxIter):
            newRoutes = self._neighbour(currentRoutes)

            if not self._allFeasible(newRoutes):
                convergence.append(bestMiles)
                T *= coolingRate
                continue

            newMiles = self._totalMiles(newRoutes)
            delta    = newMiles - currentMiles

            # Accept improving moves always; accept worse moves with Boltzmann probability
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                currentRoutes = newRoutes
                currentMiles  = newMiles

            if currentMiles < bestMiles:
                bestMiles  = currentMiles
                bestRoutes = [list(r) for r in currentRoutes]

            convergence.append(bestMiles)
            T *= coolingRate

        return bestRoutes, convergence

    def _neighbour(self, routes):
        """Relocate one random stop to a random position in a random route."""
        newRoutes = [list(r) for r in routes]
        flatStops = [
            (ri, pi)
            for ri, route in enumerate(newRoutes)
            for pi in range(len(route))
        ]

        # Need at least two stops to make a meaningful relocation
        if len(flatStops) < 2:
            return newRoutes

        ri, pi = random.choice(flatStops)
        stop   = newRoutes[ri].pop(pi)
        newRoutes = [r for r in newRoutes if r]

        # Guard against all routes becoming empty after filtering
        if not newRoutes:
            newRoutes.append([stop])
            return newRoutes

        rj  = random.randint(0, len(newRoutes) - 1)
        pos = random.randint(0, len(newRoutes[rj]))
        newRoutes[rj].insert(pos, stop)

        return newRoutes

    def _totalMiles(self, routes):
        if not routes:
            return 0
        return sum(evaluateRoute(r)["total_miles"] for r in routes)

    def _allFeasible(self, routes):
        if not routes:
            return True
        return all(evaluateRoute(r)["overall_feasible"] for r in routes)

    def _collectStats(self, routes, elapsed):
        totalMiles  = self._totalMiles(routes)
        allFeasible = self._allFeasible(routes)
        return {
            "miles":     totalMiles,
            "routes":    len(routes),
            "feasible":  allFeasible,
            "runtime_s": round(elapsed, 2),
        }