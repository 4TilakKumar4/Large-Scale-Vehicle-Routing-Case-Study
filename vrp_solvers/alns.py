"""
vrp_solvers/alns.py — Adaptive Large Neighborhood Search seeded from CW + local search.
"""

import math
import random
import time

from vrp_solvers.base import evaluateRoute, getDistance
from vrp_solvers.clarkeWright import ClarkeWrightSolver

SCORE_BEST     = 9   # new global best found
SCORE_BETTER   = 5   # better than current, accepted
SCORE_ACCEPTED = 2   # worse but accepted via SA criterion
WEIGHT_DECAY   = 0.8
UPDATE_FREQ    = 50  # iterations between weight updates


class ALNSSolver:
    """
    Adaptive Large Neighborhood Search with four destroy operators and two repair operators.
    Operator weights are updated via roulette-wheel selection every UPDATE_FREQ iterations.
    Seed is CW + consolidate + 2-opt + or-opt for a fair starting point.
    Call solve() per day; retrieve results with getStats(), getConvergence(), and getWeightHistory().
    """

    def __init__(self, maxIter=500, removeFrac=0.2, randomSeed=42):
        self.maxIter    = maxIter
        self.removeFrac = removeFrac
        self.randomSeed = randomSeed
        self._stats         = None
        self._convergence   = None
        self._weightHistory = None

    def solve(self, dayOrders):
        """Seed from CW, run ALNS, return best routes found."""
        if dayOrders.empty:
            print("  ALNSSolver: no orders for this day, skipping.")
            self._stats         = {"miles": 0, "routes": 0, "feasible": True, "runtime_s": 0.0}
            self._convergence   = []
            self._weightHistory = {}
            return []

        t0 = time.time()
        random.seed(self.randomSeed)

        seed = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True).solve(dayOrders)
        routes, convergence, weightHistory = self._search(seed)

        self._convergence   = convergence
        self._weightHistory = weightHistory
        self._stats         = self._collectStats(routes, time.time() - t0)
        return routes

    def getStats(self):
        return self._stats

    def getConvergence(self):
        return self._convergence

    def getWeightHistory(self):
        """Return destroy and repair operator weight histories from the last solve() call."""
        return self._weightHistory

    def _search(self, initRoutes):
        # Nothing to search if the seed is empty
        if not initRoutes:
            return [], [], {}

        currentRoutes = [list(r) for r in initRoutes]
        bestRoutes    = [list(r) for r in initRoutes]
        currentMiles  = self._totalMiles(currentRoutes)
        bestMiles     = currentMiles
        convergence   = [bestMiles]

        # Guard against zero initial miles causing a degenerate SA temperature
        initialTemp = currentMiles * 0.05 if currentMiles > 0 else 1.0
        T           = initialTemp
        coolingRate = 0.999

        destroyOps   = [self._randomRemoval, self._worstRemoval,
                        self._shawRemoval,   self._routeRemoval]
        repairOps    = [self._greedyInsertion, self._regretInsertion]
        destroyNames = ["random", "worst", "shaw", "route"]
        repairNames  = ["greedy", "regret"]

        dWeights = [1.0] * len(destroyOps)
        rWeights = [1.0] * len(repairOps)
        dScores  = [0.0] * len(destroyOps)
        rScores  = [0.0] * len(repairOps)
        dUses    = [0]   * len(destroyOps)
        rUses    = [0]   * len(repairOps)

        weightHistory = {
            "destroy": {n: [] for n in destroyNames},
            "repair":  {n: [] for n in repairNames},
        }

        nStops = sum(len(r) for r in currentRoutes)

        # Guard against a solution with no stops at all
        if nStops == 0:
            return bestRoutes, convergence, weightHistory

        for iteration in range(self.maxIter):
            nRemove = max(1, int(nStops * self.removeFrac))

            dIdx = random.choices(range(len(destroyOps)), weights=dWeights, k=1)[0]
            rIdx = random.choices(range(len(repairOps)),  weights=rWeights, k=1)[0]

            dUses[dIdx] += 1
            rUses[rIdx] += 1

            partial, removed = destroyOps[dIdx](currentRoutes, nRemove)
            newRoutes        = repairOps[rIdx](partial, removed)

            score = 0
            if self._allFeasible(newRoutes):
                newMiles = self._totalMiles(newRoutes)
                delta    = newMiles - currentMiles

                if newMiles < bestMiles:
                    bestMiles  = newMiles
                    bestRoutes = [list(r) for r in newRoutes]
                    score      = SCORE_BEST
                elif newMiles < currentMiles:
                    score = SCORE_BETTER
                elif random.random() < math.exp(-delta / max(T, 1e-10)):
                    score = SCORE_ACCEPTED

                if score > 0:
                    currentRoutes = newRoutes
                    currentMiles  = newMiles

            dScores[dIdx] += score
            rScores[rIdx] += score

            if (iteration + 1) % UPDATE_FREQ == 0:
                self._updateWeights(dWeights, dScores, dUses)
                self._updateWeights(rWeights, rScores, rUses)
                dScores = [0.0] * len(destroyOps)
                rScores = [0.0] * len(repairOps)
                dUses   = [0]   * len(destroyOps)
                rUses   = [0]   * len(repairOps)

            convergence.append(bestMiles)
            T *= coolingRate

            for name, w in zip(destroyNames, dWeights):
                weightHistory["destroy"][name].append(w)
            for name, w in zip(repairNames, rWeights):
                weightHistory["repair"][name].append(w)

        return bestRoutes, convergence, weightHistory

    def _updateWeights(self, weights, scores, uses):
        """Roulette-wheel weight update; clamps to a floor of 0.01 to keep all operators live."""
        for i in range(len(weights)):
            if uses[i] > 0:
                weights[i] = (WEIGHT_DECAY * weights[i]
                              + (1 - WEIGHT_DECAY) * scores[i] / uses[i])
                weights[i] = max(weights[i], 0.01)

    def _randomRemoval(self, routes, nRemove):
        """Remove nRemove stops chosen uniformly at random."""
        newRoutes = [list(r) for r in routes]
        flatStops = [(ri, pi) for ri, r in enumerate(newRoutes) for pi in range(len(r))]

        if not flatStops:
            return newRoutes, []

        # Clamp nRemove so we never ask random.sample for more items than exist
        nRemove = min(nRemove, len(flatStops))

        chosen  = random.sample(flatStops, nRemove)
        chosen.sort(reverse=True)
        removed = [newRoutes[ri].pop(pi) for ri, pi in chosen]
        newRoutes = [r for r in newRoutes if r]
        return newRoutes, removed

    def _worstRemoval(self, routes, nRemove):
        """Remove the nRemove stops that contribute most to their route's cost."""
        newRoutes = [list(r) for r in routes]
        costs     = []

        for ri, route in enumerate(newRoutes):
            # Skip single-stop routes — removing the stop would leave an empty route
            if len(route) < 2:
                continue
            baseMiles = evaluateRoute(route)["total_miles"]
            for pi in range(len(route)):
                reduced      = route[:pi] + route[pi + 1:]
                reducedMiles = evaluateRoute(reduced)["total_miles"]
                costs.append((baseMiles - reducedMiles, ri, pi))

        if not costs:
            return newRoutes, []

        costs.sort(reverse=True)
        nRemove  = min(nRemove, len(costs))
        toRemove = sorted(costs[:nRemove], key=lambda x: (x[1], x[2]), reverse=True)
        removed  = [newRoutes[ri].pop(pi) for _, ri, pi in toRemove]
        newRoutes = [r for r in newRoutes if r]
        return newRoutes, removed

    def _shawRemoval(self, routes, nRemove):
        """
        Shaw (related) removal: pick a random seed stop then remove the
        nRemove-1 geographically closest stops to encourage re-clustering.
        """
        newRoutes = [list(r) for r in routes]
        flatStops = [(ri, pi) for ri, r in enumerate(newRoutes) for pi in range(len(r))]

        if not flatStops:
            return newRoutes, []

        seedRi, seedPi = random.choice(flatStops)
        seedZip = newRoutes[seedRi][seedPi]["TOZIP"]

        others = [
            (ri, pi) for ri, pi in flatStops
            if not (ri == seedRi and pi == seedPi)
        ]
        others.sort(key=lambda x: getDistance(seedZip, newRoutes[x[0]][x[1]]["TOZIP"]))

        # Clamp so we never try to remove more stops than exist
        nRemove = min(nRemove, len(flatStops))
        chosen  = [(seedRi, seedPi)] + others[:nRemove - 1]
        chosen.sort(reverse=True)
        removed = [newRoutes[ri].pop(pi) for ri, pi in chosen]
        newRoutes = [r for r in newRoutes if r]
        return newRoutes, removed

    def _routeRemoval(self, routes, nRemove):
        """Remove all stops from the route with the fewest stops."""
        newRoutes = [list(r) for r in routes]
        if not newRoutes:
            return newRoutes, []

        targetIdx = min(range(len(newRoutes)), key=lambda i: len(newRoutes[i]))
        removed   = newRoutes.pop(targetIdx)
        return newRoutes, removed

    def _greedyInsertion(self, routes, removed):
        """Insert each removed stop at the cheapest feasible position across all routes."""
        newRoutes = [list(r) for r in routes]

        for stop in removed:
            bestCost = float("inf")
            bestRi   = -1
            bestPos  = -1

            for ri, route in enumerate(newRoutes):
                base = evaluateRoute(route)["total_miles"] if route else 0
                for pos in range(len(route) + 1):
                    trial  = route[:pos] + [stop] + route[pos:]
                    result = evaluateRoute(trial)
                    if result["overall_feasible"]:
                        cost = result["total_miles"] - base
                        if cost < bestCost:
                            bestCost = cost
                            bestRi   = ri
                            bestPos  = pos

            if bestRi >= 0:
                newRoutes[bestRi].insert(bestPos, stop)
            else:
                # No feasible position in any existing route — open a new single-stop route
                print(f"  ALNSSolver._greedyInsertion: order {int(stop['ORDERID'])} "
                      f"could not be inserted feasibly; opening new route.")
                newRoutes.append([stop])

        return newRoutes

    def _regretInsertion(self, routes, removed, k=2):
        """
        Regret-k insertion: prioritise inserting the stop whose cost difference
        between its best and k-th best positions is largest, reducing route lock-in.
        """
        newRoutes = [list(r) for r in routes]
        pending   = list(removed)

        while pending:
            regrets = []

            # Track pendingIdx so we can pop by position — stops are pandas Series,
            # and list.remove() uses == which returns a Series rather than a bool
            for pendingIdx, stop in enumerate(pending):
                insertCosts = []
                for ri, route in enumerate(newRoutes):
                    base = evaluateRoute(route)["total_miles"] if route else 0
                    for pos in range(len(route) + 1):
                        trial  = route[:pos] + [stop] + route[pos:]
                        result = evaluateRoute(trial)
                        if result["overall_feasible"]:
                            insertCosts.append((result["total_miles"] - base, ri, pos))

                insertCosts.sort(key=lambda x: x[0])

                if not insertCosts:
                    regrets.append((float("inf"), pendingIdx, stop, -1, -1))
                elif len(insertCosts) < k:
                    regrets.append((0, pendingIdx, stop, insertCosts[0][1], insertCosts[0][2]))
                else:
                    regret = insertCosts[k - 1][0] - insertCosts[0][0]
                    regrets.append((regret, pendingIdx, stop, insertCosts[0][1], insertCosts[0][2]))

            regrets.sort(key=lambda x: x[0], reverse=True)
            _, bestPendingIdx, stop, ri, pos = regrets[0]
            pending.pop(bestPendingIdx)

            if ri >= 0:
                newRoutes[ri].insert(pos, stop)
            else:
                # No feasible position found — open a new single-stop route
                print(f"  ALNSSolver._regretInsertion: order {int(stop['ORDERID'])} "
                      f"could not be inserted feasibly; opening new route.")
                newRoutes.append([stop])

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
        return {
            "miles":     self._totalMiles(routes),
            "routes":    len(routes),
            "feasible":  self._allFeasible(routes),
            "runtime_s": round(elapsed, 2),
        }