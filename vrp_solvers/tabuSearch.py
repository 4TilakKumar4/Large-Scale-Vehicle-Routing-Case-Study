"""
vrp_solvers/tabuSearch.py — Tabu Search metaheuristic seeded from CW + local search.
"""

import random
import time

from vrp_solvers.base import (
    applyLocalSearch,
    consolidateRoutes,
    evaluateRoute,
)
from vrp_solvers.clarkeWright import ClarkeWrightSolver


class TabuSearchSolver:
    """
    Tabu Search over a relocate/swap neighbourhood.
    Seed is CW + consolidate + 2-opt + or-opt for a fair starting point.
    A tabu list prevents revisiting recent moves; an aspiration criterion
    overrides the tabu status when a global best is beaten.
    Call solve() per day; retrieve results with getStats() and getConvergence().
    """

    def __init__(self, maxIter=300, tabuTenure=15, randomSeed=42):
        self.maxIter    = maxIter
        self.tabuTenure = tabuTenure
        self.randomSeed = randomSeed
        self._stats      = None
        self._convergence = None

    def solve(self, dayOrders):
        """Seed from CW, run Tabu Search, return best routes found."""
        t0 = time.time()
        random.seed(self.randomSeed)

        seed   = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True).solve(dayOrders)
        routes, convergence = self._search(seed)

        self._convergence = convergence
        self._stats       = self._collectStats(routes, time.time() - t0)
        return routes

    def getStats(self):
        return self._stats

    def getConvergence(self):
        return self._convergence

    def _search(self, initRoutes):
        currentRoutes = [list(r) for r in initRoutes]
        bestRoutes    = [list(r) for r in initRoutes]
        bestMiles     = self._totalMiles(bestRoutes)
        tabuList      = {}
        convergence   = [bestMiles]

        for iteration in range(self.maxIter):
            moves    = self._generateMoves(currentRoutes)
            random.shuffle(moves)

            bestMoveMiles  = float("inf")
            bestMove       = None
            bestMoveRoutes = None

            for move in moves[:200]:
                moveKey   = (move[0], move[1], move[2])
                newRoutes = self._applyMove(currentRoutes, move)
                if newRoutes is None:
                    continue

                newMiles     = self._totalMiles(newRoutes)
                isTabu       = tabuList.get(moveKey, 0) > iteration
                aspirationOk = newMiles < bestMiles

                if (not isTabu or aspirationOk) and newMiles < bestMoveMiles:
                    bestMoveMiles  = newMiles
                    bestMove       = move
                    bestMoveRoutes = newRoutes

            if bestMove is None:
                convergence.append(bestMiles)
                continue

            currentRoutes             = bestMoveRoutes
            moveKey                   = (bestMove[0], bestMove[1], bestMove[2])
            tabuList[moveKey]         = iteration + self.tabuTenure

            if bestMoveMiles < bestMiles:
                bestMiles  = bestMoveMiles
                bestRoutes = [list(r) for r in currentRoutes]

            convergence.append(bestMiles)

        return bestRoutes, convergence

    def _generateMoves(self, routes):
        """Generate all single-stop relocations and cross-route swaps."""
        moves = []

        for ri, route in enumerate(routes):
            for pi in range(len(route)):
                for rj in range(len(routes)):
                    if ri == rj:
                        continue
                    for pos in range(len(routes[rj]) + 1):
                        moves.append(("relocate", ri, pi, rj, pos))

        for ri in range(len(routes)):
            for pi in range(len(routes[ri])):
                for rj in range(ri + 1, len(routes)):
                    for pj in range(len(routes[rj])):
                        moves.append(("swap", ri, pi, rj, pj))

        return moves

    def _applyMove(self, routes, move):
        """Apply a relocate or swap move; return new routes or None if infeasible."""
        newRoutes = [list(r) for r in routes]

        if move[0] == "relocate":
            _, ri, pi, rj, pos = move
            stop = newRoutes[ri].pop(pi)
            newRoutes[rj].insert(pos, stop)
            newRoutes = [r for r in newRoutes if r]

        elif move[0] == "swap":
            _, ri, pi, rj, pj = move
            newRoutes[ri][pi], newRoutes[rj][pj] = newRoutes[rj][pj], newRoutes[ri][pi]

        for r in newRoutes:
            if not evaluateRoute(r)["overall_feasible"]:
                return None

        return newRoutes

    def _totalMiles(self, routes):
        return sum(evaluateRoute(r)["total_miles"] for r in routes)

    def _collectStats(self, routes, elapsed):
        totalMiles  = self._totalMiles(routes)
        allFeasible = all(evaluateRoute(r)["overall_feasible"] for r in routes)
        return {
            "miles":     totalMiles,
            "routes":    len(routes),
            "feasible":  allFeasible,
            "runtime_s": round(elapsed, 2),
        }
