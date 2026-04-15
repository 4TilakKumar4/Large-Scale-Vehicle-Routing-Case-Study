"""
vrp_solvers/mixedFleetSolver.py — Mixed-fleet CVRP solver (Vans + Straight Trucks).

Fleet rules:
  - Orders where straight_truck_required == "yes" must go on a Straight Truck.
  - Orders whose cube exceeds ST_CAPACITY must go on a Van.
  - All remaining orders are flexible and may go on either vehicle type.

Algorithm:
  1. Unified Clarke-Wright over all orders simultaneously so flexible orders
     cluster geographically with nearby ST-required orders before fleet
     assignment is finalised.
  2. Intra-fleet 2-opt + or-opt improvement per fleet.
  3. Cross-fleet improvement: move flexible stops between fleets when cheaper.

Call solve() per day; retrieve results with getStats() and getConvergence().
"""

import time
from itertools import combinations

from vrp_solvers.base import (
    DAYS,
    DEPOT_ZIP,
    DRIVING_SPEED,
    MAX_DRIVING,
    MAX_DUTY,
    ST_CAPACITY,
    VAN_CAPACITY,
    WINDOW_OPEN,
    evaluateMixedRoute,
    getDistance,
)

def _getMiles(route, vehicleType):
    """Return total route miles. Feasibility is tracked separately."""
    if not route:
        return 0.0
    return evaluateMixedRoute(route, vehicleType)["total_miles"]

def _isStRequired(stop):
    return str(stop.get("straight_truck_required", "no")).strip().lower() == "yes"

def _isTooLargeForST(stop):
    return float(stop["CUBE"]) > ST_CAPACITY

def _isFlexible(stop):
    return not _isStRequired(stop) and not _isTooLargeForST(stop)

class MixedFleetSolver:
    """
    Mixed-fleet solver: Vans (3,200 ft³) and Straight Trucks (1,400 ft³).
    ST-required orders are locked to Straight Trucks.
    Orders too large for ST are locked to Vans.
    All others are flexible and assigned to the cheaper fleet.
    Call solve() per day; retrieve results with getStats() and getConvergence().
    """

    def __init__(self):
        self._stats     = None
        self._vanRoutes = None
        self._stRoutes  = None

    def solve(self, dayOrders):
        """Run unified CW, improve each fleet, apply cross-fleet moves; return all routes."""
        if dayOrders.empty:
            print("  MixedFleetSolver: no orders for this day, skipping.")
            self._vanRoutes = []
            self._stRoutes  = []
            self._stats     = {"miles": 0, "routes": 0, "feasible": True,
                               "runtime_s": 0.0, "van_routes": 0, "st_routes": 0}
            return []

        t0 = time.time()

        stops           = list(dayOrders.to_dict("records"))
        vanRoutes, stRoutes = self._unifiedCW(stops)

        vanRoutes = self._improveFleet(vanRoutes, "van")
        stRoutes  = self._improveFleet(stRoutes,  "st")
        vanRoutes, stRoutes = self._crossFleetImprove(vanRoutes, stRoutes)

        self._vanRoutes = vanRoutes
        self._stRoutes  = stRoutes
        self._stats     = self._collectStats(vanRoutes, stRoutes, time.time() - t0)

        return vanRoutes + stRoutes

    def getStats(self):
        return self._stats

    def getConvergence(self):
        return None

    def getWeightHistory(self):
        return None

    def getVanRoutes(self):
        return self._vanRoutes or []

    def getStRoutes(self):
        return self._stRoutes or []

    # Construction

    def _unifiedCW(self, stops):
        """
        Run Clarke-Wright over all day stops simultaneously.
        Fleet type is tracked per route and governs merge eligibility:
          - "st"   : contains at least one ST-required stop
          - "van"  : contains at least one stop too large for ST
          - "flex" : all stops are flexible; can still merge with either fleet
        Routes cannot merge if one is "st" and the other is "van".
        """
        routes    = [[s] for s in stops]
        routeCube = [float(s["CUBE"]) for s in stops]
        routeVeh  = []
        routeOf   = {}

        for i, s in enumerate(stops):
            oid = int(s["ORDERID"])
            routeOf[oid] = i
            if _isStRequired(s):
                routeVeh.append("st")
            elif _isTooLargeForST(s):
                routeVeh.append("van")
            else:
                routeVeh.append("flex")

        # Savings: s(i,j) = d(depot,i) + d(depot,j) - d(i,j)
        savings = []
        for i, j in combinations(range(len(stops)), 2):
            zi = stops[i]["TOZIP"]
            zj = stops[j]["TOZIP"]
            s  = (getDistance(DEPOT_ZIP, zi)
                  + getDistance(DEPOT_ZIP, zj)
                  - getDistance(zi, zj))
            savings.append((s, i, j))
            savings.append((s, j, i))
        savings.sort(key=lambda x: -x[0])

        for sVal, iIdx, jIdx in savings:
            if sVal <= 0:
                break

            oidI = int(stops[iIdx]["ORDERID"])
            oidJ = int(stops[jIdx]["ORDERID"])
            ri   = routeOf[oidI]
            rj   = routeOf[oidJ]

            if ri == rj or routes[ri] is None or routes[rj] is None:
                continue

            # Only merge tail-of-ri to head-of-rj
            if (int(routes[ri][-1]["ORDERID"]) != oidI or
                    int(routes[rj][0]["ORDERID"])  != oidJ):
                continue

            vi = routeVeh[ri]
            vj = routeVeh[rj]

            # Block van↔st merges
            if (vi == "st" and vj == "van") or (vi == "van" and vj == "st"):
                continue

            mergedVeh = ("st"   if (vi == "st"   or vj == "st")   else
                         "van"  if (vi == "van"  or vj == "van")  else
                         "flex")
            checkVeh  = "van" if mergedVeh in ("van", "flex") else "st"
            cap       = VAN_CAPACITY if checkVeh == "van" else ST_CAPACITY

            if routeCube[ri] + routeCube[rj] > cap:
                continue

            merged = routes[ri] + routes[rj]
            if not evaluateMixedRoute(merged, checkVeh)["overall_feasible"]:
                continue

            routes[ri]     = merged
            routeCube[ri] += routeCube[rj]
            routeVeh[ri]   = mergedVeh
            routes[rj]     = None

            for stop in merged:
                routeOf[int(stop["ORDERID"])] = ri

        vanRoutes = [r for i, r in enumerate(routes)
                     if r is not None and routeVeh[i] != "st"]
        stRoutes  = [r for i, r in enumerate(routes)
                     if r is not None and routeVeh[i] == "st"]

        return vanRoutes, stRoutes

    # Improvement operators

    def _twoOpt(self, route, vehicleType):
        """Intra-route 2-opt improvement."""
        if len(route) < 3:
            return route
        best     = route[:]
        bestMi   = _getMiles(best, vehicleType)
        improved = True

        while improved:
            improved = False
            for i in range(len(best) - 1):
                for j in range(i + 2, len(best)):
                    cand = best[:i + 1] + best[i + 1:j + 1][::-1] + best[j + 1:]
                    res  = evaluateMixedRoute(cand, vehicleType)
                    if res["overall_feasible"] and res["total_miles"] < bestMi - 0.01:
                        best, bestMi, improved = cand, res["total_miles"], True
                        break
                if improved:
                    break

        return best

    def _orOpt(self, routes, vehicleType):
        """Inter-route or-opt: relocate stops between routes of the same fleet."""
        cap      = VAN_CAPACITY if vehicleType == "van" else ST_CAPACITY
        routes   = [r[:] for r in routes]
        improved = True

        while improved:
            improved = False
            for r1 in range(len(routes)):
                if not routes[r1]:
                    continue
                for pos in range(len(routes[r1])):
                    stop  = routes[r1][pos]
                    newR1 = routes[r1][:pos] + routes[r1][pos + 1:]

                    for r2 in range(len(routes)):
                        if not routes[r2]:
                            continue
                        cubeR2 = sum(float(s["CUBE"]) for s in routes[r2])
                        if r1 != r2 and cubeR2 + float(stop["CUBE"]) > cap:
                            continue

                        before = _getMiles(routes[r1], vehicleType)
                        if r1 != r2:
                            before += _getMiles(routes[r2], vehicleType)

                        for ins in range(len(routes[r2]) + 1):
                            if r1 == r2:
                                cand = newR1[:ins] + [stop] + newR1[ins:]
                                res  = evaluateMixedRoute(cand, vehicleType)
                                if res["overall_feasible"] and res["total_miles"] < _getMiles(routes[r1], vehicleType) - 0.01:
                                    routes[r1] = cand
                                    improved   = True
                                    break
                            else:
                                newR2 = routes[r2][:ins] + [stop] + routes[r2][ins:]
                                # newR1 may be empty if the only stop was relocated
                                r1r = ({"overall_feasible": True, "total_miles": 0}
                                       if not newR1 else
                                       evaluateMixedRoute(newR1, vehicleType))
                                r2r = evaluateMixedRoute(newR2, vehicleType)
                                if r1r["overall_feasible"] and r2r["overall_feasible"]:
                                    if r1r["total_miles"] + r2r["total_miles"] < before - 0.01:
                                        routes[r1] = newR1
                                        routes[r2] = newR2
                                        improved   = True
                                        break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        return [r for r in routes if r]

    def _improveFleet(self, routes, vehicleType):
        """Apply 2-opt then or-opt to a fleet's routes."""
        improved = [self._twoOpt(r, vehicleType) for r in routes]
        return self._orOpt(improved, vehicleType)

    def _crossFleetImprove(self, vanRoutes, stRoutes):
        """
        Move flexible stops between fleets when doing so reduces total miles.
        ST-required stops and stops too large for ST are never moved.
        """
        improved = True

        while improved:
            improved = False

            # Try moving flexible stops from Van → ST
            for vIdx in range(len(vanRoutes)):
                if not vanRoutes[vIdx]:
                    continue
                for pos in range(len(vanRoutes[vIdx])):
                    stop = vanRoutes[vIdx][pos]
                    if not _isFlexible(stop):
                        continue

                    newVan = vanRoutes[vIdx][:pos] + vanRoutes[vIdx][pos + 1:]

                    for sIdx in range(len(stRoutes)):
                        if not stRoutes[sIdx]:
                            continue
                        cubeS = sum(float(s["CUBE"]) for s in stRoutes[sIdx])
                        if cubeS + float(stop["CUBE"]) > ST_CAPACITY:
                            continue

                        before = (_getMiles(vanRoutes[vIdx], "van")
                                  + _getMiles(stRoutes[sIdx], "st"))

                        for ins in range(len(stRoutes[sIdx]) + 1):
                            newSt = stRoutes[sIdx][:ins] + [stop] + stRoutes[sIdx][ins:]
                            vr    = evaluateMixedRoute(newVan, "van") if newVan else {"overall_feasible": True, "total_miles": 0}
                            sr    = evaluateMixedRoute(newSt,  "st")
                            if vr["overall_feasible"] and sr["overall_feasible"]:
                                if vr["total_miles"] + sr["total_miles"] < before - 0.01:
                                    vanRoutes[vIdx] = newVan
                                    stRoutes[sIdx]  = newSt
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            if improved:
                continue

            # Try moving flexible stops from ST → Van
            for sIdx in range(len(stRoutes)):
                if not stRoutes[sIdx]:
                    continue
                for pos in range(len(stRoutes[sIdx])):
                    stop = stRoutes[sIdx][pos]
                    if _isStRequired(stop):
                        continue  # ST-required stays on ST

                    newSt = stRoutes[sIdx][:pos] + stRoutes[sIdx][pos + 1:]

                    for vIdx in range(len(vanRoutes)):
                        if not vanRoutes[vIdx]:
                            continue
                        cubeV = sum(float(s["CUBE"]) for s in vanRoutes[vIdx])
                        if cubeV + float(stop["CUBE"]) > VAN_CAPACITY:
                            continue

                        before = (_getMiles(stRoutes[sIdx],  "st")
                                  + _getMiles(vanRoutes[vIdx], "van"))

                        for ins in range(len(vanRoutes[vIdx]) + 1):
                            newVan = vanRoutes[vIdx][:ins] + [stop] + vanRoutes[vIdx][ins:]
                            sr     = evaluateMixedRoute(newSt,  "st") if newSt else {"overall_feasible": True, "total_miles": 0}
                            vr     = evaluateMixedRoute(newVan, "van")
                            if sr["overall_feasible"] and vr["overall_feasible"]:
                                if sr["total_miles"] + vr["total_miles"] < before - 0.01:
                                    stRoutes[sIdx]  = newSt
                                    vanRoutes[vIdx] = newVan
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        return [r for r in vanRoutes if r], [r for r in stRoutes if r]

    # Stats

    def _collectStats(self, vanRoutes, stRoutes, elapsed):
        vanMiles    = sum(_getMiles(r, "van") for r in vanRoutes)
        stMiles     = sum(_getMiles(r, "st")  for r in stRoutes)
        totalMiles  = vanMiles + stMiles
        totalRoutes = len(vanRoutes) + len(stRoutes)
        allFeasible = (
            all(evaluateMixedRoute(r, "van")["overall_feasible"] for r in vanRoutes)
            and all(evaluateMixedRoute(r, "st")["overall_feasible"]  for r in stRoutes)
        )

        return {
            "miles":      int(totalMiles),
            "routes":     totalRoutes,
            "feasible":   allFeasible,
            "runtime_s":  round(elapsed, 2),
            "van_miles":  int(vanMiles),
            "st_miles":   int(stMiles),
            "van_routes": len(vanRoutes),
            "st_routes":  len(stRoutes),
        }


class ALNSMixedFleetSolver:
    """
    ALNS for the mixed-fleet CVRP (Vans + Straight Trucks).

    Routes are stored as tagged dicts {"type": "van"|"st", "stops": [...]}.
    This unified representation lets the ALNS destroy/repair loop move flexible
    stops between fleets — the key improvement over the CW-based MixedFleetSolver.

    Fleet assignment rules are enforced at the repair step:
      - ST-required stops  → ST routes only
      - Too-large stops    → Van routes only
      - Flexible stops     → whichever fleet produces a cheaper feasible insertion

    Seed: MixedFleetSolver (CW + local search + cross-fleet improvement).
    Call solve() per day; retrieve results with getStats(), getVanRoutes(), getStRoutes().
    """

    def __init__(self, maxIter=500, removeFrac=0.2, randomSeed=42):
        self.maxIter    = maxIter
        self.removeFrac = removeFrac
        self.randomSeed = randomSeed
        self._vanRoutes     = []
        self._stRoutes      = []
        self._stats         = None
        self._convergence   = None
        self._weightHistory = None

    def solve(self, dayOrders):
        if dayOrders.empty:
            print("  ALNSMixedFleetSolver: no orders for this day, skipping.")
            self._vanRoutes     = []
            self._stRoutes      = []
            self._stats         = {"miles": 0, "routes": 0, "feasible": True,
                                   "runtime_s": 0.0, "van_routes": 0, "st_routes": 0,
                                   "van_miles": 0, "st_miles": 0}
            self._convergence   = []
            self._weightHistory = {}
            return []

        t0 = time.time()
        import random as _random
        _random.seed(self.randomSeed)

        # Seed from MixedFleetSolver
        seed = MixedFleetSolver()
        seed.solve(dayOrders)

        tagged = (
            [{"type": "van", "stops": list(r)} for r in seed.getVanRoutes()]
            + [{"type": "st",  "stops": list(r)} for r in seed.getStRoutes()]
        )

        best, convergence, weightHistory = self._search(tagged)

        vanRoutes = [r["stops"] for r in best if r["type"] == "van"]
        stRoutes  = [r["stops"] for r in best if r["type"] == "st"]

        self._vanRoutes     = vanRoutes
        self._stRoutes      = stRoutes
        self._convergence   = convergence
        self._weightHistory = weightHistory
        self._stats         = self._collectStats(vanRoutes, stRoutes, time.time() - t0)
        return vanRoutes + stRoutes

    def getStats(self):
        return self._stats

    def getConvergence(self):
        return self._convergence

    def getWeightHistory(self):
        return self._weightHistory

    def getVanRoutes(self):
        return self._vanRoutes or []

    def getStRoutes(self):
        return self._stRoutes or []

    def _search(self, initTagged):
        import random as _random, math as _math

        if not initTagged:
            return [], [], {}

        current  = [{"type": r["type"], "stops": list(r["stops"])} for r in initTagged]
        best     = [{"type": r["type"], "stops": list(r["stops"])} for r in initTagged]
        curMiles = self._totalMiles(current)
        bestMiles = curMiles

        convergence = [bestMiles]
        T = curMiles * 0.05 if curMiles > 0 else 1.0
        coolingRate = 0.999

        destroyOps   = [self._randomRemoval, self._worstRemoval,
                        self._shawRemoval,   self._routeRemoval]
        repairOps    = [self._greedyInsertion, self._regretInsertion]
        destroyNames = ["random", "worst", "shaw", "route"]
        repairNames  = ["greedy", "regret"]

        dWeights = [1.0] * 4
        rWeights = [1.0] * 2
        dScores  = [0.0] * 4
        rScores  = [0.0] * 2
        dUses    = [0]   * 4
        rUses    = [0]   * 2

        weightHistory = {
            "destroy": {n: [] for n in destroyNames},
            "repair":  {n: [] for n in repairNames},
        }

        nStops = sum(len(r["stops"]) for r in current)
        if nStops == 0:
            return best, convergence, weightHistory

        UPDATE_FREQ = 50
        SCORE_BEST = 9; SCORE_BETTER = 5; SCORE_ACCEPTED = 2
        WEIGHT_DECAY = 0.8

        for iteration in range(self.maxIter):
            nRemove = max(1, int(nStops * self.removeFrac))

            dIdx = _random.choices(range(4), weights=dWeights, k=1)[0]
            rIdx = _random.choices(range(2), weights=rWeights, k=1)[0]
            dUses[dIdx] += 1
            rUses[rIdx] += 1

            partial, removed = destroyOps[dIdx](current, nRemove)
            newTagged        = repairOps[rIdx](partial, removed)

            score = 0
            if self._allFeasible(newTagged):
                newMiles = self._totalMiles(newTagged)
                delta    = newMiles - curMiles

                if newMiles < bestMiles:
                    bestMiles = newMiles
                    best      = [{"type": r["type"], "stops": list(r["stops"])}
                                 for r in newTagged]
                    score     = SCORE_BEST
                elif newMiles < curMiles:
                    score = SCORE_BETTER
                elif _random.random() < _math.exp(-delta / max(T, 1e-10)):
                    score = SCORE_ACCEPTED

                if score > 0:
                    current  = newTagged
                    curMiles = newMiles

            dScores[dIdx] += score
            rScores[rIdx] += score

            if (iteration + 1) % UPDATE_FREQ == 0:
                for i in range(4):
                    if dUses[i] > 0:
                        dWeights[i] = max(WEIGHT_DECAY * dWeights[i]
                                          + (1 - WEIGHT_DECAY) * dScores[i] / dUses[i], 0.01)
                for i in range(2):
                    if rUses[i] > 0:
                        rWeights[i] = max(WEIGHT_DECAY * rWeights[i]
                                          + (1 - WEIGHT_DECAY) * rScores[i] / rUses[i], 0.01)
                dScores = [0.0]*4; rScores = [0.0]*2
                dUses   = [0]*4;   rUses   = [0]*2

            convergence.append(bestMiles)
            T *= coolingRate

            for name, w in zip(destroyNames, dWeights):
                weightHistory["destroy"][name].append(w)
            for name, w in zip(repairNames, rWeights):
                weightHistory["repair"][name].append(w)

        return best, convergence, weightHistory

    def _totalMiles(self, tagged):
        total = 0
        for r in tagged:
            if r["stops"]:
                total += evaluateMixedRoute(r["stops"], r["type"])["total_miles"]
        return total

    def _allFeasible(self, tagged):
        return all(
            evaluateMixedRoute(r["stops"], r["type"])["overall_feasible"]
            for r in tagged if r["stops"]
        )

    def _stopCategory(self, stop):
        """Classify a stop for fleet-aware repair."""
        if _isStRequired(stop):
            return "st_required"
        if _isTooLargeForST(stop):
            return "too_large"
        return "flexible"

    def _randomRemoval(self, tagged, nRemove):
        import random as _random
        current  = [{"type": r["type"], "stops": list(r["stops"])} for r in tagged]
        flatStops = [(ri, pi) for ri, r in enumerate(current)
                     for pi in range(len(r["stops"]))]
        if not flatStops:
            return current, []
        nRemove = min(nRemove, len(flatStops))
        chosen  = sorted(_random.sample(flatStops, nRemove), reverse=True)
        removed = [current[ri]["stops"].pop(pi) for ri, pi in chosen]
        return [r for r in current if r["stops"]], removed

    def _worstRemoval(self, tagged, nRemove):
        current = [{"type": r["type"], "stops": list(r["stops"])} for r in tagged]
        costs   = []
        for ri, r in enumerate(current):
            if len(r["stops"]) < 2:
                continue
            base = evaluateMixedRoute(r["stops"], r["type"])["total_miles"]
            for pi in range(len(r["stops"])):
                reduced      = r["stops"][:pi] + r["stops"][pi+1:]
                reducedMiles = evaluateMixedRoute(reduced, r["type"])["total_miles"]
                costs.append((base - reducedMiles, ri, pi))
        if not costs:
            return current, []
        costs.sort(reverse=True)
        nRemove  = min(nRemove, len(costs))
        toRemove = sorted(costs[:nRemove], key=lambda x: (x[1], x[2]), reverse=True)
        removed  = [current[ri]["stops"].pop(pi) for _, ri, pi in toRemove]
        return [r for r in current if r["stops"]], removed

    def _shawRemoval(self, tagged, nRemove):
        import random as _random
        current   = [{"type": r["type"], "stops": list(r["stops"])} for r in tagged]
        flatStops = [(ri, pi) for ri, r in enumerate(current)
                     for pi in range(len(r["stops"]))]
        if not flatStops:
            return current, []
        seedRi, seedPi = _random.choice(flatStops)
        seedZip = current[seedRi]["stops"][seedPi]["TOZIP"]
        others  = [(ri, pi) for ri, pi in flatStops
                   if not (ri == seedRi and pi == seedPi)]
        others.sort(key=lambda x: getDistance(seedZip, current[x[0]]["stops"][x[1]]["TOZIP"]))
        nRemove = min(nRemove, len(flatStops))
        chosen  = sorted([(seedRi, seedPi)] + others[:nRemove-1], reverse=True)
        removed = [current[ri]["stops"].pop(pi) for ri, pi in chosen]
        return [r for r in current if r["stops"]], removed

    def _routeRemoval(self, tagged, nRemove):
        current = [{"type": r["type"], "stops": list(r["stops"])} for r in tagged]
        if not current:
            return current, []
        targetIdx = min(range(len(current)), key=lambda i: len(current[i]["stops"]))
        removed   = current.pop(targetIdx)["stops"]
        return current, removed

    def _greedyInsertion(self, tagged, removed):
        current = [{"type": r["type"], "stops": list(r["stops"])} for r in tagged]
        for stop in removed:
            cat      = self._stopCategory(stop)
            bestCost = float("inf")
            bestRi   = -1
            bestPos  = -1
            bestType = None

            for ri, r in enumerate(current):
                if cat == "st_required" and r["type"] != "st":
                    continue
                if cat == "too_large" and r["type"] != "van":
                    continue
                base = evaluateMixedRoute(r["stops"], r["type"])["total_miles"] if r["stops"] else 0
                for pos in range(len(r["stops"]) + 1):
                    trial  = r["stops"][:pos] + [stop] + r["stops"][pos:]
                    result = evaluateMixedRoute(trial, r["type"])
                    if result["overall_feasible"]:
                        cost = result["total_miles"] - base
                        if cost < bestCost:
                            bestCost = cost; bestRi = ri; bestPos = pos; bestType = r["type"]

            # Flexible stops: also try opposite fleet type
            if cat == "flexible":
                altType = "st"
                for ri, r in enumerate(current):
                    if r["type"] != altType:
                        continue
                    base = evaluateMixedRoute(r["stops"], altType)["total_miles"] if r["stops"] else 0
                    for pos in range(len(r["stops"]) + 1):
                        trial  = r["stops"][:pos] + [stop] + r["stops"][pos:]
                        result = evaluateMixedRoute(trial, altType)
                        if result["overall_feasible"]:
                            cost = result["total_miles"] - base
                            if cost < bestCost:
                                bestCost = cost; bestRi = ri; bestPos = pos; bestType = altType

            if bestRi >= 0:
                current[bestRi]["stops"].insert(bestPos, stop)
            else:
                # Open new route of the appropriate type
                newType = "st" if cat == "st_required" else "van"
                current.append({"type": newType, "stops": [stop]})

        return current

    def _regretInsertion(self, tagged, removed):
        current = [{"type": r["type"], "stops": list(r["stops"])} for r in tagged]
        pending = list(removed)

        while pending:
            regrets = []
            for pIdx, stop in enumerate(pending):
                cat         = self._stopCategory(stop)
                insertCosts = []

                for ri, r in enumerate(current):
                    if cat == "st_required" and r["type"] != "st":
                        continue
                    if cat == "too_large" and r["type"] != "van":
                        continue
                    fleets = [r["type"]]
                    if cat == "flexible":
                        fleets = [r["type"]]
                    base = evaluateMixedRoute(r["stops"], r["type"])["total_miles"] if r["stops"] else 0
                    for pos in range(len(r["stops"]) + 1):
                        trial  = r["stops"][:pos] + [stop] + r["stops"][pos:]
                        result = evaluateMixedRoute(trial, r["type"])
                        if result["overall_feasible"]:
                            insertCosts.append((result["total_miles"] - base, ri, pos))

                insertCosts.sort(key=lambda x: x[0])

                if not insertCosts:
                    regrets.append((float("inf"), pIdx, stop, -1, -1))
                elif len(insertCosts) < 2:
                    regrets.append((0, pIdx, stop, insertCosts[0][1], insertCosts[0][2]))
                else:
                    regret = insertCosts[1][0] - insertCosts[0][0]
                    regrets.append((regret, pIdx, stop, insertCosts[0][1], insertCosts[0][2]))

            regrets.sort(key=lambda x: x[0], reverse=True)
            _, bestPIdx, stop, ri, pos = regrets[0]
            pending.pop(bestPIdx)

            if ri >= 0:
                current[ri]["stops"].insert(pos, stop)
            else:
                cat     = self._stopCategory(stop)
                newType = "st" if cat == "st_required" else "van"
                current.append({"type": newType, "stops": [stop]})

        return current

    def _collectStats(self, vanRoutes, stRoutes, elapsed):
        from vrp_solvers.base import evaluateMixedRoute as _emr
        vanMiles    = sum(_emr(r, "van")["total_miles"] for r in vanRoutes)
        stMiles     = sum(_emr(r, "st")["total_miles"]  for r in stRoutes)
        totalMiles  = vanMiles + stMiles
        totalRoutes = len(vanRoutes) + len(stRoutes)
        allFeasible = (
            all(_emr(r, "van")["overall_feasible"] for r in vanRoutes)
            and all(_emr(r, "st")["overall_feasible"]  for r in stRoutes)
        )
        return {
            "miles":      int(totalMiles),
            "routes":     totalRoutes,
            "feasible":   allFeasible,
            "runtime_s":  round(elapsed, 2),
            "van_miles":  int(vanMiles),
            "st_miles":   int(stMiles),
            "van_routes": len(vanRoutes),
            "st_routes":  len(stRoutes),
        }