
import os
import csv
import time
import colorsys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

DELIVERIES_FILE = os.path.join(BASE_DIR, "deliveries.xlsx")
DISTANCES_FILE  = os.path.join(BASE_DIR, "distances.xlsx")

VAN_CAP        = 3200
ST_CAP         = 1400
UNLOAD_VAN     = 0.030
UNLOAD_ST      = 0.043   # ST lift-gate unloading is slower per ft³
MIN_UNLOAD     = 30
SPEED_MPH      = 40
WINDOW_OPEN    = 480     # 8 am in minutes from midnight
WINDOW_CLOSE   = 1080    # 6 pm
MAX_DRIVE_MIN  = 660     # 11-hour DOT drive cap
MAX_DUTY_MIN   = 840     # 14-hour DOT duty cap
BREAK_MIN      = 600     # mandatory 10-hour overnight rest
WEEKS_PER_YEAR = 52

DAYS         = ["Mon", "Tue", "Wed", "Thu", "Fri"]
Q1_WEEKLY_MI = 7025.0

DAY_COLORS = {
    "Mon": {"van": "#e53935", "st": "#ff8a80"},
    "Tue": {"van": "#fb8c00", "st": "#ffd180"},
    "Wed": {"van": "#1e88e5", "st": "#82b1ff"},
    "Thu": {"van": "#8e24aa", "st": "#ea80fc"},
    "Fri": {"van": "#43a047", "st": "#b9f6ca"},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def loadData():
    """Return all lookup structures needed by the solver."""
    orders   = pd.read_excel(DELIVERIES_FILE, sheet_name="OrderTable")
    location = pd.read_excel(DELIVERIES_FILE, sheet_name="LocationTable")
    distRaw  = pd.read_excel(DISTANCES_FILE,  sheet_name="Sheet1", header=None)

    orders = orders[orders["ORDERID"] != 0].copy()
    orders["CUBE"]   = orders["CUBE"].astype(int)
    orders["TOZIP"]  = orders["TOZIP"].astype(str).str.zfill(5)
    orders["ST_REQ"] = orders["ST required?"].str.lower() == "yes"

    location = location.dropna(subset=["ZIPID"]).copy()
    location["ZIPID"]   = location["ZIPID"].astype(int)
    location["ZIP_STR"] = location["ZIP"].astype(int).astype(str).str.zfill(5)
    location = location.rename(columns={"Y": "LAT", "X": "LON"})

    zipToId      = dict(zip(location["ZIP_STR"], location["ZIPID"]))
    zipidToCoord = {int(r["ZIPID"]): (float(r["LAT"]), float(r["LON"]))
                    for _, r in location.iterrows()}
    zipidToCity  = {int(r["ZIPID"]): str(r["CITY"])  for _, r in location.iterrows()}
    zipidToState = {int(r["ZIPID"]): str(r["STATE"]) for _, r in location.iterrows()}

    depotId  = int(zipToId["01887"])
    colIds   = distRaw.iloc[1, 2:].astype(int).tolist()
    distArr  = distRaw.iloc[2:, 2:].astype(float).values
    distIdx  = {int(z): i for i, z in enumerate(colIds)}

    orderDict = {}
    for _, row in orders.iterrows():
        oid = int(row["ORDERID"])
        orderDict[oid] = {
            "cube":  int(row["CUBE"]),
            "day":   row["DayOfWeek"],
            "zipId": int(zipToId[row["TOZIP"]]),
            "stReq": bool(row["ST_REQ"]),
        }

    return orderDict, depotId, distArr, distIdx, zipidToCoord, zipidToCity, zipidToState


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def getDist(a, b, distArr, distIdx):
    return distArr[distIdx[int(a)], distIdx[int(b)]]


def driveMins(a, b, distArr, distIdx):
    return (getDist(a, b, distArr, distIdx) / SPEED_MPH) * 60


def unloadMins(cube, vehicle):
    rate = UNLOAD_VAN if vehicle == "van" else UNLOAD_ST
    return max(MIN_UNLOAD, rate * cube)


def getCap(vehicle):
    return VAN_CAP if vehicle == "van" else ST_CAP


# ---------------------------------------------------------------------------
# Feasibility checker
# ---------------------------------------------------------------------------

def simulate(route, vehicle, orderDict, depotId, distArr, distIdx):
    """Return feasibility and route miles. No overnight — DOT excess = infeasible."""
    cap = getCap(vehicle)
    if not route:
        return {"feasible": True, "miles": 0.0, "overnight": False,
                "driveD1": 0.0, "dutyD1": 0.0, "driveD2": 0.0, "dutyD2": 0.0}
    if sum(orderDict[o]["cube"] for o in route) > cap:
        return {"feasible": False, "miles": 0.0, "overnight": False}

    firstZip = orderDict[route[0]]["zipId"]
    clock    = WINDOW_OPEN - driveMins(depotId, firstZip, distArr, distIdx)
    loc      = depotId
    miles    = driveAcc = dutyAcc = 0.0

    for oid in route:
        dest   = orderDict[oid]["zipId"]
        legMi  = getDist(loc, dest, distArr, distIdx)
        legMin = driveMins(loc, dest, distArr, distIdx)
        ul     = unloadMins(orderDict[oid]["cube"], vehicle)

        if legMin > MAX_DRIVE_MIN - driveAcc or legMin > MAX_DUTY_MIN - dutyAcc:
            return {"feasible": False, "miles": miles, "overnight": False}

        clock    += legMin; driveAcc += legMin; dutyAcc += legMin
        miles    += legMi;  loc = dest
        if clock > WINDOW_CLOSE:
            return {"feasible": False, "miles": miles, "overnight": False}
        clock += ul; dutyAcc += ul

    retMi  = getDist(loc, depotId, distArr, distIdx)
    retMin = driveMins(loc, depotId, distArr, distIdx)
    if driveAcc + retMin > MAX_DRIVE_MIN or dutyAcc + retMin > MAX_DUTY_MIN:
        return {"feasible": False, "miles": miles, "overnight": False}

    miles += retMi; driveAcc += retMin; dutyAcc += retMin
    return {"feasible": True, "miles": miles, "overnight": False,
            "driveD1": driveAcc / 60, "dutyD1": dutyAcc / 60,
            "driveD2": 0.0, "dutyD2": 0.0}


def getMiles(route, vehicle, orderDict, depotId, distArr, distIdx):
    res = simulate(route, vehicle, orderDict, depotId, distArr, distIdx)
    return res["miles"] if res["feasible"] else float("inf")


# ---------------------------------------------------------------------------
# Improvement heuristics
# ---------------------------------------------------------------------------

def twoOpt(route, vehicle, orderDict, depotId, distArr, distIdx):
    """Reverse sub-segments until no reversal reduces route miles."""
    if len(route) < 3:
        return route
    best     = route[:]
    bestMi   = getMiles(best, vehicle, orderDict, depotId, distArr, distIdx)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                cand = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                res  = simulate(cand, vehicle, orderDict, depotId, distArr, distIdx)
                if res["feasible"] and res["miles"] < bestMi - 0.01:
                    best, bestMi, improved = cand, res["miles"], True
                    break
            if improved:
                break
    return best


def orOpt(routes, vehicle, orderDict, depotId, distArr, distIdx):
    """Relocate stops between same-fleet routes — catches misassignments 2-opt cannot fix."""
    cap      = getCap(vehicle)
    routes   = [r[:] for r in routes]
    improved = True
    while improved:
        improved = False
        for r1 in range(len(routes)):
            if not routes[r1]: continue
            for pos in range(len(routes[r1])):
                oid   = routes[r1][pos]
                newR1 = routes[r1][:pos] + routes[r1][pos+1:]
                for r2 in range(len(routes)):
                    if not routes[r2]: continue
                    cubeR2 = sum(orderDict[o]["cube"] for o in routes[r2])
                    if r1 != r2 and cubeR2 + orderDict[oid]["cube"] > cap: continue
                    before = getMiles(routes[r1], vehicle, orderDict, depotId, distArr, distIdx)
                    if r1 != r2:
                        before += getMiles(routes[r2], vehicle, orderDict, depotId, distArr, distIdx)
                    for ins in range(len(routes[r2]) + 1):
                        if r1 == r2:
                            cand = newR1[:ins] + [oid] + newR1[ins:]
                            res  = simulate(cand, vehicle, orderDict, depotId, distArr, distIdx)
                            if res["feasible"] and res["miles"] < getMiles(routes[r1], vehicle, orderDict, depotId, distArr, distIdx) - 0.01:
                                routes[r1] = cand; improved = True; break
                        else:
                            newR2 = routes[r2][:ins] + [oid] + routes[r2][ins:]
                            r1r   = simulate(newR1, vehicle, orderDict, depotId, distArr, distIdx)
                            r2r   = simulate(newR2, vehicle, orderDict, depotId, distArr, distIdx)
                            if r1r["feasible"] and r2r["feasible"]:
                                if r1r["miles"] + r2r["miles"] < before - 0.01:
                                    routes[r1] = newR1; routes[r2] = newR2
                                    improved = True; break
                    if improved: break
                if improved: break
            if improved: break
    return [r for r in routes if r]


# ---------------------------------------------------------------------------
# Construction heuristic
# ---------------------------------------------------------------------------

def unifiedCW(dayOids, orderDict, depotId, distArr, distIdx):
    """Clarke-Wright over all orders at once so flexible orders cluster with nearby ST routes."""
    routes    = [[oid] for oid in dayOids]
    routeCube = [orderDict[oid]["cube"] for oid in dayOids]
    routeOf   = {oid: i for i, oid in enumerate(dayOids)}

    routeVeh = []
    for oid in dayOids:
        if orderDict[oid]["stReq"]:       routeVeh.append("st")
        elif orderDict[oid]["cube"] > ST_CAP: routeVeh.append("van")
        else:                              routeVeh.append("flex")

    savings = []
    for i, j in combinations(dayOids, 2):
        s = (getDist(depotId, orderDict[i]["zipId"], distArr, distIdx)
             + getDist(depotId, orderDict[j]["zipId"], distArr, distIdx)
             - getDist(orderDict[i]["zipId"], orderDict[j]["zipId"], distArr, distIdx))
        savings.append((s, i, j))
        savings.append((s, j, i))
    savings.sort(key=lambda x: -x[0])

    for sVal, iOid, jOid in savings:
        if sVal <= 0: break
        ri = routeOf[iOid]; rj = routeOf[jOid]
        if ri == rj or routes[ri] is None or routes[rj] is None: continue
        if routes[ri][-1] != iOid or routes[rj][0] != jOid:     continue
        vi = routeVeh[ri]; vj = routeVeh[rj]
        if (vi == "st" and vj == "van") or (vi == "van" and vj == "st"): continue
        mergedVeh = ("st"  if (vi == "st"  or vj == "st")  else
                     "van" if (vi == "van" or vj == "van") else "flex")
        checkVeh  = "st" if mergedVeh == "st" else "van"
        if routeCube[ri] + routeCube[rj] > getCap(checkVeh): continue
        merged = routes[ri] + routes[rj]
        if not simulate(merged, checkVeh, orderDict, depotId, distArr, distIdx)["feasible"]: continue
        routes[ri]     = merged
        routeCube[ri] += routeCube[rj]
        routeVeh[ri]   = mergedVeh
        routes[rj]     = None
        for oid in routes[ri]: routeOf[oid] = ri

    vanRoutes = [r for i, r in enumerate(routes) if r is not None and routeVeh[i] != "st"]
    stRoutes  = [r for i, r in enumerate(routes) if r is not None and routeVeh[i] == "st"]
    return vanRoutes, stRoutes


def crossFleetImprove(vanRoutes, stRoutes, orderDict, depotId, distArr, distIdx):
    """Move flexible stops between fleets when cheaper. ST-required orders never move."""
    improved = True
    while improved:
        improved = False
        for vIdx in range(len(vanRoutes)):
            if not vanRoutes[vIdx]: continue
            for pos in range(len(vanRoutes[vIdx])):
                oid = vanRoutes[vIdx][pos]
                if orderDict[oid]["stReq"] or orderDict[oid]["cube"] > ST_CAP: continue
                newVan = vanRoutes[vIdx][:pos] + vanRoutes[vIdx][pos+1:]
                for sIdx in range(len(stRoutes)):
                    if not stRoutes[sIdx]: continue
                    if sum(orderDict[o]["cube"] for o in stRoutes[sIdx]) + orderDict[oid]["cube"] > ST_CAP: continue
                    before = (getMiles(vanRoutes[vIdx], "van", orderDict, depotId, distArr, distIdx)
                              + getMiles(stRoutes[sIdx],  "st",  orderDict, depotId, distArr, distIdx))
                    for ins in range(len(stRoutes[sIdx]) + 1):
                        newSt = stRoutes[sIdx][:ins] + [oid] + stRoutes[sIdx][ins:]
                        vr    = simulate(newVan, "van", orderDict, depotId, distArr, distIdx)
                        sr    = simulate(newSt,  "st",  orderDict, depotId, distArr, distIdx)
                        if vr["feasible"] and sr["feasible"] and vr["miles"] + sr["miles"] < before - 0.01:
                            vanRoutes[vIdx] = newVan; stRoutes[sIdx] = newSt
                            improved = True; break
                    if improved: break
                if improved: break
            if improved: break
        if improved: continue
        for sIdx in range(len(stRoutes)):
            if not stRoutes[sIdx]: continue
            for pos in range(len(stRoutes[sIdx])):
                oid = stRoutes[sIdx][pos]
                if orderDict[oid]["stReq"]: continue
                newSt = stRoutes[sIdx][:pos] + stRoutes[sIdx][pos+1:]
                for vIdx in range(len(vanRoutes)):
                    if not vanRoutes[vIdx]: continue
                    if sum(orderDict[o]["cube"] for o in vanRoutes[vIdx]) + orderDict[oid]["cube"] > VAN_CAP: continue
                    before = (getMiles(stRoutes[sIdx],  "st",  orderDict, depotId, distArr, distIdx)
                              + getMiles(vanRoutes[vIdx], "van", orderDict, depotId, distArr, distIdx))
                    for ins in range(len(vanRoutes[vIdx]) + 1):
                        newVan = vanRoutes[vIdx][:ins] + [oid] + vanRoutes[vIdx][ins:]
                        sr     = simulate(newSt,  "st",  orderDict, depotId, distArr, distIdx)
                        vr     = simulate(newVan, "van", orderDict, depotId, distArr, distIdx)
                        if sr["feasible"] and vr["feasible"] and sr["miles"] + vr["miles"] < before - 0.01:
                            stRoutes[sIdx] = newSt; vanRoutes[vIdx] = newVan
                            improved = True; break
                    if improved: break
                if improved: break
            if improved: break
    return [r for r in vanRoutes if r], [r for r in stRoutes if r]


# ---------------------------------------------------------------------------
# Day solver
# ---------------------------------------------------------------------------

def solveDay(day, ordersByDay, orderDict, depotId, distArr, distIdx):
    """Run unified CW, improve each fleet, then correct cross-fleet assignments."""
    dayOids             = ordersByDay[day]
    vanRoutes, stRoutes = unifiedCW(dayOids, orderDict, depotId, distArr, distIdx)
    vanRoutes = orOpt(
        [twoOpt(r, "van", orderDict, depotId, distArr, distIdx) for r in vanRoutes],
        "van", orderDict, depotId, distArr, distIdx)
    stRoutes  = orOpt(
        [twoOpt(r, "st",  orderDict, depotId, distArr, distIdx) for r in stRoutes],
        "st",  orderDict, depotId, distArr, distIdx)
    vanRoutes, stRoutes = crossFleetImprove(
        vanRoutes, stRoutes, orderDict, depotId, distArr, distIdx)
    vanMi = sum(getMiles(r, "van", orderDict, depotId, distArr, distIdx) for r in vanRoutes)
    stMi  = sum(getMiles(r, "st",  orderDict, depotId, distArr, distIdx) for r in stRoutes)
    return {"vanRoutes": vanRoutes, "stRoutes": stRoutes,
            "vanMiles":  vanMi,    "stMiles":  stMi,
            "total":     vanMi + stMi,
            "nVan":      len(vanRoutes), "nSt": len(stRoutes)}


# ---------------------------------------------------------------------------
# Excel solver verification
# ---------------------------------------------------------------------------

def verifySolution(allVan, allSt, orderDict, depotId, distArr, distIdx):
    """Mirror all five Excel SolutionCheck constraints and print per-route audit."""
    print()
    print(f"  {'─'*78}")
    print(f"  SOLUTION VERIFICATION  (mirrors Sol_Copy_Sr26.xlsm SolutionCheck)")
    print(f"  {'─'*78}")
    print(f"  {'Route':<14} {'Veh':<4} {'Stops':>5} {'Miles':>7} "
          f"{'D1 Drive':>9} {'D1 Duty':>8} {'D2 Drive':>9} {'D2 Duty':>8} "
          f"{'Cube':>6} {'Cap':>5}  Status")
    print(f"  {'-'*78}")

    allServed  = []
    violations = []

    for day in DAYS:
        for ri, route in enumerate(allVan[day]):
            res  = simulate(route, "van", orderDict, depotId, distArr, distIdx)
            cube = sum(orderDict[o]["cube"] for o in route)
            tag  = f"{day} Van R{ri+1}"
            iss  = _routeIssues(res, cube, "van")
            if iss: violations.append((tag, iss))
            _printRow(tag, "Van", len(route), res, cube, VAN_CAP, iss)
            allServed.extend(route)

        for ri, route in enumerate(allSt[day]):
            res  = simulate(route, "st", orderDict, depotId, distArr, distIdx)
            cube = sum(orderDict[o]["cube"] for o in route)
            tag  = f"{day} ST  R{ri+1}"
            iss  = _routeIssues(res, cube, "st")
            if iss: violations.append((tag, iss))
            _printRow(tag, "ST", len(route), res, cube, ST_CAP, iss)
            allServed.extend(route)

    print(f"  {'-'*78}")

    orderOk  = len(allServed) == 261 and len(set(allServed)) == 261
    allPass  = orderOk and not violations

    checks = [
        ("261 orders served, none duplicated",     orderOk),
        ("Van capacity ≤ 3,200 ft³ per route",
         all(sum(orderDict[o]["cube"] for o in r) <= VAN_CAP for d in DAYS for r in allVan[d])),
        ("ST capacity  ≤ 1,400 ft³ per route",
         all(sum(orderDict[o]["cube"] for o in r) <= ST_CAP  for d in DAYS for r in allSt[d])),
        ("DOT drive ≤ 11 h per shift",             not any("DRIVE" in i for _, il in violations for i in il)),
        ("DOT duty  ≤ 14 h per shift",             not any("DUTY"  in i for _, il in violations for i in il)),
        ("All routes feasible",                    not violations),
    ]
    print()
    print(f"  {'Constraint':<48} Status")
    print(f"  {'-'*48} {'-'*8}")
    for desc, ok in checks:
        print(f"  {desc:<48} {'✓  PASS' if ok else '✗  FAIL'}")
    print(f"  {'─'*58}")
    print(f"  {'✓  ALL CONSTRAINTS SATISFIED' if allPass else '✗  VIOLATIONS FOUND'}")
    if violations:
        for tag, iss in violations:
            print(f"     {tag}: {', '.join(iss)}")
    print(f"  {'─'*58}")
    return allPass


def _routeIssues(res, cube, vehicle):
    issues = []
    if not res["feasible"]:                issues.append("INFEASIBLE")
    if cube > getCap(vehicle):             issues.append(f"CAP {cube}>{getCap(vehicle)}")
    if res.get("driveD1", 0) > 11.01:     issues.append(f"D1-DRIVE {res['driveD1']:.2f}h")
    if res.get("dutyD1",  0) > 14.01:     issues.append(f"D1-DUTY {res['dutyD1']:.2f}h")
    if res.get("driveD2", 0) > 11.01:     issues.append(f"D2-DRIVE {res['driveD2']:.2f}h")
    if res.get("dutyD2",  0) > 14.01:     issues.append(f"D2-DUTY {res['dutyD2']:.2f}h")
    return issues


def _printRow(tag, veh, stops, res, cube, cap, issues):
    d1d = res.get("driveD1", 0); d1y = res.get("dutyD1", 0)
    d2d = res.get("driveD2", 0); d2y = res.get("dutyD2", 0)
    status = "✓ OK" if not issues else "✗ FAIL"
    print(f"  {tag:<14} {veh:<4} {stops:>5} {res['miles']:>7.0f} "
          f"{d1d:>8.2f}h {d1y:>7.2f}h {d2d:>8.2f}h {d2y:>7.2f}h "
          f"{cube:>6} {cap:>5}  {status}")


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def exportSolverInput(allVan, allSt, outPath):
    """Write flat 0-separated OID list for paste into Sol_Copy_Sr26.xlsm column B."""
    lines = []
    for day in DAYS:
        for route in allVan[day] + allSt[day]:
            lines.append("0")
            lines.extend(str(o) for o in route)
    lines.append("0")
    with open(outPath, "w") as f:
        f.write("\n".join(lines))


def exportCsv(allVan, allSt, orderDict, depotId, distArr, distIdx,
              zipidToCity, zipidToState, outPath):
    """Write one CSV row per route with full OID sequence and location detail."""
    def cityState(oid):
        zid = orderDict[oid]["zipId"]
        return f"{zipidToCity.get(zid,'?')}, {zipidToState.get(zid,'?')}"

    def oidSeq(route):
        return "0 -> " + " -> ".join(str(o) for o in route) + " -> 0"

    def locSeq(route):
        parts = ["0 (DC-Wilmington,MA)"]
        for o in route:
            parts.append(f"{o} ({cityState(o)})")
        parts.append("0 (DC-Wilmington,MA)")
        return " -> ".join(parts)

    rows = []
    for day in DAYS:
        for ri, route in enumerate(allVan[day]):
            res  = simulate(route, "van", orderDict, depotId, distArr, distIdx)
            cube = sum(orderDict[o]["cube"] for o in route)
            rows.append({
                "Route":          f"{day}_Van_R{ri+1}",
                "Day":            day,
                "Vehicle":        "Van",
                "Stops":          len(route),
                "Miles":          round(res["miles"], 1),
                "Cube_ft3":       cube,
                "Capacity_ft3":   VAN_CAP,
                "Util_Pct":       round(cube / VAN_CAP * 100, 1),
                "Overnight":      "Yes" if res["overnight"] else "No",
                "D1_Drive_h":     round(res.get("driveD1", 0), 2),
                "D1_Duty_h":      round(res.get("dutyD1",  0), 2),
                "D2_Drive_h":     round(res.get("driveD2", 0), 2),
                "D2_Duty_h":      round(res.get("dutyD2",  0), 2),
                "Start":          "DC-Wilmington,MA",
                "End":            "DC-Wilmington,MA",
                "First_Stop":     cityState(route[0]),
                "Last_Stop":      cityState(route[-1]),
                "Order_IDs":      oidSeq(route),
                "Route_Sequence": locSeq(route),
            })
        for ri, route in enumerate(allSt[day]):
            res  = simulate(route, "st", orderDict, depotId, distArr, distIdx)
            cube = sum(orderDict[o]["cube"] for o in route)
            rows.append({
                "Route":          f"{day}_ST_R{ri+1}",
                "Day":            day,
                "Vehicle":        "StraightTruck",
                "Stops":          len(route),
                "Miles":          round(res["miles"], 1),
                "Cube_ft3":       cube,
                "Capacity_ft3":   ST_CAP,
                "Util_Pct":       round(cube / ST_CAP * 100, 1),
                "Overnight":      "Yes" if res["overnight"] else "No",
                "D1_Drive_h":     round(res.get("driveD1", 0), 2),
                "D1_Duty_h":      round(res.get("dutyD1",  0), 2),
                "D2_Drive_h":     round(res.get("driveD2", 0), 2),
                "D2_Duty_h":      round(res.get("dutyD2",  0), 2),
                "Start":          "DC-Wilmington,MA",
                "End":            "DC-Wilmington,MA",
                "First_Stop":     cityState(route[0]),
                "Last_Stop":      cityState(route[-1]),
                "Order_IDs":      oidSeq(route),
                "Route_Sequence": locSeq(route),
            })

    with open(outPath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plotFleetAnalysis(allVan, allSt, orderDict, depotId, distArr, distIdx, outPath):
    """Stacked daily miles and average capacity utilisation per fleet."""
    vanMi   = [sum(getMiles(r, "van", orderDict, depotId, distArr, distIdx) for r in allVan[d]) for d in DAYS]
    stMi    = [sum(getMiles(r, "st",  orderDict, depotId, distArr, distIdx) for r in allSt[d])  for d in DAYS]
    vanUtil = [sum(orderDict[o]["cube"] for r in allVan[d] for o in r) / max(1, len(allVan[d]) * VAN_CAP) * 100 for d in DAYS]
    stUtil  = [sum(orderDict[o]["cube"] for r in allSt[d]  for o in r) / max(1, len(allSt[d])  * ST_CAP)  * 100 for d in DAYS]
    nVan    = sum(len(allVan[d]) for d in DAYS)
    nSt     = sum(len(allSt[d])  for d in DAYS)

    vanColor = "#4db8e8"; stColor = "#f0a030"
    x        = np.arange(len(DAYS))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f1117")

    ax = axes[0]; ax.set_facecolor("#0f1117")
    ax.bar(x, vanMi, 0.5, label=f"Van ({nVan})", color=vanColor, edgecolor="white", lw=0.5, zorder=3)
    ax.bar(x, stMi,  0.5, bottom=vanMi, label=f"ST ({nSt})",  color=stColor,  edgecolor="white", lw=0.5, zorder=3)
    for i, (v, s) in enumerate(zip(vanMi, stMi)):
        ax.text(i, v + s + 15, f"{v+s:.0f}", ha="center", color="white", fontsize=9.5, fontweight="bold")
        if v > 50: ax.text(i, v / 2,     f"{v:.0f}", ha="center", color="white",   fontsize=8)
        if s > 50: ax.text(i, v + s / 2, f"{s:.0f}", ha="center", color="#0f1117", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(DAYS)
    ax.set_ylabel("Miles", color="white")
    ax.set_title("Daily Miles — Van vs Straight Truck", color="white", fontweight="bold")
    ax.legend(facecolor="#1e2130", labelcolor="white", fontsize=9)
    ax.tick_params(colors="white"); ax.yaxis.grid(True, color="#333", ls="--", zorder=0)
    for sp in ax.spines.values(): sp.set_color("#444")

    ax2 = axes[1]; ax2.set_facecolor("#0f1117")
    ax2.bar(x - 0.13, vanUtil, 0.26, label="Van %", color=vanColor, edgecolor="white", lw=0.5, zorder=3)
    ax2.bar(x + 0.13, stUtil,  0.26, label="ST %",  color=stColor,  edgecolor="white", lw=0.5, zorder=3)
    ax2.axhline(100, color="#e05252", ls="--", lw=1.5, label="100% cap", zorder=5)
    for i, (v, s) in enumerate(zip(vanUtil, stUtil)):
        ax2.text(i - 0.13, v + 1.5, f"{v:.0f}%", ha="center", color=vanColor, fontsize=8, fontweight="bold")
        ax2.text(i + 0.13, s + 1.5, f"{s:.0f}%", ha="center", color=stColor,  fontsize=8, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(DAYS); ax2.set_ylim(0, 118)
    ax2.set_title("Avg Capacity Utilisation %", color="white", fontweight="bold")
    ax2.set_ylabel("Utilisation %", color="white")
    ax2.legend(facecolor="#1e2130", labelcolor="white", fontsize=9)
    ax2.tick_params(colors="white"); ax2.yaxis.grid(True, color="#333", ls="--", zorder=0)
    for sp in ax2.spines.values(): sp.set_color("#444")

    fig.suptitle("Q2 (No Overnight) — Fleet Analysis", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outPath, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()


def plotRouteMap(allVan, allSt, orderDict, depotId, distArr, distIdx,
                 zipidToCoord, weeklyTotal, outPath):
    """2×5 static grid — Van (top row) and ST (bottom row) routes per day."""
    def mkShade(hexC, idx, total):
        h, s, v = colorsys.rgb_to_hsv(int(hexC[1:3], 16) / 255,
                                       int(hexC[3:5], 16) / 255,
                                       int(hexC[5:7], 16) / 255)
        v2 = 0.4 + idx / max(1, total - 1) * 0.45
        r, g, b = colorsys.hsv_to_rgb(h, s, v2)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    depotCoord = zipidToCoord[depotId]
    fig, axes  = plt.subplots(2, 5, figsize=(26, 11))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        f"Q2 (No Overnight) Route Map  ·  Van top, ST bottom  ·  dashed = overnight  ·  "
        f"Weekly: {weeklyTotal:,.0f} mi  ·  Annual: {weeklyTotal*52:,.0f} mi",
        color="white", fontsize=11, fontweight="bold", y=1.01)

    for col, day in enumerate(DAYS):
        for row, (routeList, vehicle, baseHex) in enumerate([
            (allVan[day], "van", DAY_COLORS[day]["van"]),
            (allSt[day],  "st",  DAY_COLORS[day]["st"]),
        ]):
            ax    = axes[row][col]; ax.set_facecolor("#0d1b2a")
            n     = len(routeList)
            dayMi = sum(getMiles(r, vehicle, orderDict, depotId, distArr, distIdx) for r in routeList)
            onCt  = sum(1 for r in routeList
                        if simulate(r, vehicle, orderDict, depotId, distArr, distIdx)["overnight"])

            for ri, route in enumerate(routeList):
                c    = mkShade(baseHex, ri, n)
                isOn = simulate(route, vehicle, orderDict, depotId, distArr, distIdx)["overnight"]
                path = [depotId] + [orderDict[o]["zipId"] for o in route] + [depotId]
                lons = [zipidToCoord[z][1] for z in path if z in zipidToCoord]
                lats = [zipidToCoord[z][0] for z in path if z in zipidToCoord]
                ax.plot(lons, lats, "--" if isOn else "-", color=c, lw=2.0, zorder=3, alpha=0.9)
                for seq, oid in enumerate(route):
                    coord = zipidToCoord.get(orderDict[oid]["zipId"])
                    if not coord: continue
                    ax.scatter(coord[1], coord[0], s=32, color=c,
                               marker="o" if vehicle == "van" else "s", zorder=4)
                    ax.text(coord[1], coord[0], str(seq + 1),
                            ha="center", va="center", fontsize=4.5,
                            color="white", fontweight="bold", zorder=5)
                ax.plot([], [], color=c, lw=2, ls="--" if isOn else "-", label=f"R{ri+1}")

            ax.scatter(depotCoord[1], depotCoord[0], s=220, color="#f5a623", marker="*", zorder=6)
            ax.text(depotCoord[1], depotCoord[0] + 0.06, "DC",
                    ha="center", fontsize=7, color="#f5a623", fontweight="bold", zorder=7)
            vl  = "Van" if vehicle == "van" else "ST"
            ttl = f"{day} {vl}  ·  {n} routes  ·  {dayMi:.0f} mi"
            if onCt: ttl += f"  ({onCt} overnight)"
            ax.set_title(ttl, color="white", fontsize=8.5, fontweight="bold", pad=4)
            ax.tick_params(colors="#445", labelsize=5.5)
            for sp in ax.spines.values(): sp.set_color("#223")
            if col == 0:
                ax.set_ylabel("Van Routes" if row == 0 else "ST Routes", color="#aaa", fontsize=8)
            ax.legend(loc="lower right", fontsize=5, framealpha=0.6,
                      facecolor="#0d1b2a", labelcolor="white", edgecolor="#334")

    plt.tight_layout()
    plt.savefig(outPath, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    (orderDict, depotId, distArr, distIdx,
     zipidToCoord, zipidToCity, zipidToState) = loadData()

    ordersByDay = {d: [oid for oid, o in orderDict.items() if o["day"] == d]
                   for d in DAYS}

    print("=" * 60)
    print("  Q2 — MIXED FLEET NO OVERNIGHT")
    print("=" * 60)

    allVan      = {}; allSt = {}
    weeklyTotal = 0.0; daySummary = []

    for day in DAYS:
        print(f"  Solving {day}...", end=" ", flush=True)
        t0  = time.time()
        res = solveDay(day, ordersByDay, orderDict, depotId, distArr, distIdx)
        print(f"done ({time.time()-t0:.1f}s)  "
              f"Van {res['nVan']}×{res['vanMiles']:.0f}mi  "
              f"ST {res['nSt']}×{res['stMiles']:.0f}mi")
        allVan[day]  = res["vanRoutes"]; allSt[day] = res["stRoutes"]
        weeklyTotal += res["total"];     daySummary.append(res)

    nVan = sum(s["nVan"] for s in daySummary)
    nSt  = sum(s["nSt"]  for s in daySummary)

    print(f"\n  Weekly : {weeklyTotal:>10,.0f} mi")
    print(f"  Annual : {weeklyTotal*52:>10,.0f} mi")
    print(f"  vs Q1  : {(weeklyTotal - Q1_WEEKLY_MI)/Q1_WEEKLY_MI*100:>+10.1f}%")
    print(f"  Routes : Van {nVan}  ST {nSt}")

    verifySolution(allVan, allSt, orderDict, depotId, distArr, distIdx)

    print("\n  Exporting CSV...")
    exportCsv(allVan, allSt, orderDict, depotId, distArr, distIdx,
              zipidToCity, zipidToState,
              os.path.join(OUTPUT_DIR, "Q2_NOON_Routes.csv"))

    print("  Writing solver input...")
    exportSolverInput(allVan, allSt, os.path.join(OUTPUT_DIR, "Q2_NOON_SolverInput.txt"))

    print("  Plotting fleet analysis...")
    plotFleetAnalysis(allVan, allSt, orderDict, depotId, distArr, distIdx,
                      os.path.join(OUTPUT_DIR, "Q2_NOON_Viz1_Fleet.png"))

    print("  Plotting route map...")
    plotRouteMap(allVan, allSt, orderDict, depotId, distArr, distIdx,
                 zipidToCoord, weeklyTotal,
                 os.path.join(OUTPUT_DIR, "Q2_NOON_Viz2_RouteMap.png"))

    print(f"\n  All outputs written to: {OUTPUT_DIR}")
    print(f"  Paste Q2_NOON_SolverInput.txt into column B of Sol_Copy_Sr26.xlsm")
    print("=" * 60)


if __name__ == "__main__":
    main()