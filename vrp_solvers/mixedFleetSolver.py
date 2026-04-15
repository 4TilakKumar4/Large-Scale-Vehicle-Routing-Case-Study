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