import os
import math
import warnings
from itertools import combinations

import pandas as pd
import folium
from folium import plugins

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VAN_CAPACITY  = 3200
UNLOAD_RATE   = 0.03   # min / cubic foot
MIN_TIME      = 30     # minutes
DRIVING_SPEED = 40     # mph
WINDOW_OPEN   = 8
WINDOW_CLOSE  = 18
BREAK_TIME    = 10     # hours
MAX_DRIVING   = 11     # hours
MAX_DUTY      = 14     # hours
DEPOT_ZIP     = 1887

DAYS        = ["Mon", "Tue", "Wed", "Thu", "Fri"]
ROUTE_CACHE = {}

OUTPUT_FILE = os.path.join(BASE_DIR, "relaxed_routes_map.html")

DAY_COLORS = {
    "Mon": "#E63946",
    "Tue": "#F4A261",
    "Wed": "#2A9D8F",
    "Thu": "#457B9D",
    "Fri": "#9B5DE5",
}


def loadInputs():
    """Read orders and distance matrix from Excel; coerce column types."""
    orders = pd.read_excel(os.path.join(BASE_DIR, "deliveries.xlsx"), sheet_name="OrderTable")
    distances = pd.read_excel(os.path.join(BASE_DIR, "distances.xlsx"), sheet_name="Sheet1")

    orders = orders[orders["ORDERID"] != 0].copy()
    orders["CUBE"] = pd.to_numeric(orders["CUBE"], errors="raise")
    orders["FROMZIP"] = pd.to_numeric(orders["FROMZIP"], errors="raise")
    orders["TOZIP"] = pd.to_numeric(orders["TOZIP"], errors="raise")
    orders["ORDERID"] = pd.to_numeric(orders["ORDERID"], errors="raise")

    distances = distances.rename(columns={"Unnamed: 0": "ZIP", "Unnamed: 1": "ZIPID"})
    distances = distances[distances["ZIP"] != "Zip"].copy()
    distances["ZIP"] = pd.to_numeric(distances["ZIP"], errors="raise")
    distances["ZIPID"] = pd.to_numeric(distances["ZIPID"], errors="raise")
    distances = distances.set_index("ZIP")

    distMatrix = distances.drop(columns=["ZIPID"]).copy()
    distMatrix.columns = pd.to_numeric(distMatrix.columns, errors="coerce")

    return orders, distMatrix


def loadZipCoords():
    """
    Read ZIP coordinates from the LocationTable sheet in deliveries.xlsx.
    In this file:
      X = longitude
      Y = latitude
    """
    locations = pd.read_excel(os.path.join(BASE_DIR, "deliveries.xlsx"), sheet_name="LocationTable")

    locations["ZIP"] = pd.to_numeric(locations["ZIP"], errors="coerce")
    locations["X"] = pd.to_numeric(locations["X"], errors="coerce")
    locations["Y"] = pd.to_numeric(locations["Y"], errors="coerce")
    locations = locations.dropna(subset=["ZIP", "X", "Y"])

    zipCoords = {}
    for _, row in locations.iterrows():
        zipCode = int(row["ZIP"])
        lon = float(row["X"])
        lat = float(row["Y"])
        zipCoords[zipCode] = (lat, lon)

    return zipCoords


def getDistance(zip1, zip2):
    return DIST_MATRIX.loc[zip1, zip2]


def getUnloadTime(cube):
    """Unload time in hours, enforcing a minimum floor of MIN_TIME minutes."""
    return max(MIN_TIME, UNLOAD_RATE * cube) / 60.0


def toClock(hours):
    """Convert fractional hours to HH:MM string."""
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h += 1
        m = 0
    return f"{h:02d}:{m:02d}"


def routeIds(route):
    return [int(stop["ORDERID"]) for stop in route]


def routeKey(route):
    return tuple(int(stop["ORDERID"]) for stop in route)


def getAngleFromDepot(zipCode):
    """
    Polar angle of a ZIP relative to the depot, measured from the depot coordinates.
    """
    if DEPOT_ZIP not in ZIP_COORDS:
        raise KeyError(f"Depot ZIP {DEPOT_ZIP} not found in ZIP_COORDS")

    zipCode = int(zipCode)
    if zipCode not in ZIP_COORDS:
        raise KeyError(f"ZIP {zipCode} not found in ZIP_COORDS")

    depotLat, depotLon = ZIP_COORDS[DEPOT_ZIP]
    lat, lon = ZIP_COORDS[zipCode]

    angle = math.atan2(lat - depotLat, lon - depotLon)
    if angle < 0:
        angle += 2 * math.pi

    return angle


def evaluateRoute(routeList, verbose=False):
    """
    Simulate the route and return feasibility flags and cost metrics.
    Checks three hard constraints: capacity, delivery windows, and DOT HOS.
    """
    if len(routeList) == 0:
        return {
            "total_miles":       0,
            "total_drive":       0.0,
            "total_unload":      0.0,
            "total_wait":        0.0,
            "total_duty":        0.0,
            "total_cube":        0,
            "return_time":       0.0,
            "capacity_feasible": True,
            "window_feasible":   True,
            "dot_feasible":      True,
            "overall_feasible":  True,
        }

    cacheKey = None
    if not verbose:
        cacheKey = routeKey(routeList)
        cached = ROUTE_CACHE.get(cacheKey)
        if cached is not None:
            return cached

    firstZip = routeList[0]["TOZIP"]
    firstDrive = getDistance(DEPOT_ZIP, firstZip) / DRIVING_SPEED
    dispatchTime = max(0.0, WINDOW_OPEN - firstDrive)

    currentZip = DEPOT_ZIP
    currentTime = dispatchTime

    totalMiles = 0
    totalDrive = 0
    totalUnload = 0
    totalWait = 0
    totalCube = 0
    windowFeasible = True

    if verbose:
        print("Dispatch:", toClock(dispatchTime))

    for stop in routeList:
        stopZip = stop["TOZIP"]
        cube = stop["CUBE"]

        milesLeg = getDistance(currentZip, stopZip)
        drive = milesLeg / DRIVING_SPEED
        arrival = currentTime + drive
        serviceStart = max(arrival, WINDOW_OPEN)
        wait = max(0.0, WINDOW_OPEN - arrival)
        unload = getUnloadTime(cube)
        departure = serviceStart + unload

        timeOfDay = serviceStart % 24
        endOfService = departure % 24
        beforeClose = (WINDOW_OPEN <= timeOfDay <= WINDOW_CLOSE) and (endOfService <= WINDOW_CLOSE)
        windowFeasible = windowFeasible and beforeClose

        if verbose:
            print(
                f"  Stop {int(stop['ORDERID'])}: arrive {toClock(arrival)} | "
                f"start {toClock(serviceStart)} | depart {toClock(departure)} | "
                f"ok={beforeClose}"
            )

        totalMiles += milesLeg
        totalDrive += drive
        totalWait += wait
        totalUnload += unload
        totalCube += cube

        currentTime = departure
        currentZip = stopZip

    milesBack = getDistance(currentZip, DEPOT_ZIP)
    driveBack = milesBack / DRIVING_SPEED
    returnTime = currentTime + driveBack

    totalMiles += milesBack
    totalDrive += driveBack
    totalDuty = totalDrive + totalUnload + totalWait

    capacityFeasible = totalCube <= VAN_CAPACITY
    dotFeasible = totalDrive <= MAX_DRIVING and totalDuty <= MAX_DUTY
    overallFeasible = capacityFeasible and windowFeasible and dotFeasible

    results = {
        "total_miles":       int(totalMiles),
        "total_drive":       round(float(totalDrive), 3),
        "total_unload":      round(float(totalUnload), 3),
        "total_wait":        round(float(totalWait), 3),
        "total_duty":        round(float(totalDuty), 3),
        "total_cube":        int(totalCube),
        "return_time":       round(float(returnTime), 3),
        "capacity_feasible": bool(capacityFeasible),
        "window_feasible":   bool(windowFeasible),
        "dot_feasible":      bool(dotFeasible),
        "overall_feasible":  bool(overallFeasible),
    }

    if not verbose:
        if len(ROUTE_CACHE) > 300000:
            ROUTE_CACHE.clear()
        ROUTE_CACHE[cacheKey] = results

    return results


def computeSavings(ordersDF):
    """Clarke-Wright savings s(i,j) = d(depot,i) + d(depot,j) - d(i,j), sorted descending."""
    orderList = ordersDF.to_dict("records")
    savings = []

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


def clarkeWright(dayOrders):
    """
    Parallel savings construction heuristic.
    Each stop starts as its own route; pairs are merged in descending savings order when feasible.
    """
    orderRecords = {int(row["ORDERID"]): row for _, row in dayOrders.iterrows()}

    routes = {oid: [rec] for oid, rec in orderRecords.items()}
    routeOf = {oid: oid for oid in orderRecords}
    headOf = {oid: oid for oid in orderRecords}
    tailOf = {oid: oid for oid in orderRecords}

    for s, oidI, oidJ in computeSavings(dayOrders):
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
            candidates.append((routeI + routeJ, ridI, ridJ))
        if tailOf[ridJ] == oidJ and headOf[ridI] == oidI:
            candidates.append((routeJ + routeI, ridJ, ridI))
        if tailOf[ridI] == oidI and tailOf[ridJ] == oidJ:
            candidates.append((routeI + routeJ[::-1], ridI, ridJ))
        if headOf[ridI] == oidI and headOf[ridJ] == oidJ:
            candidates.append((routeI[::-1] + routeJ, ridI, ridJ))

        for mergedRoute, keepRid, dropRid in candidates:
            if evaluateRoute(mergedRoute, verbose=False)["overall_feasible"]:
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


def bestRelocation(stop, routes, skipRouteIdx):
    """Find the cheapest feasible insertion of stop across all routes except skipRouteIdx."""
    bestTargetIdx = None
    bestNewRoute = None
    bestExtraMiles = float("inf")

    for j, route in enumerate(routes):
        if j == skipRouteIdx:
            continue

        baseMiles = evaluateRoute(route, verbose=False)["total_miles"]

        for pos in range(len(route) + 1):
            trialRoute = route[:pos] + [stop] + route[pos:]
            results = evaluateRoute(trialRoute, verbose=False)

            if results["overall_feasible"]:
                extraMiles = results["total_miles"] - baseMiles
                if extraMiles < bestExtraMiles:
                    bestTargetIdx = j
                    bestNewRoute = trialRoute
                    bestExtraMiles = extraMiles

    return bestTargetIdx, bestNewRoute


def tryEliminateOneRoute(allRoutes):
    """
    Try to remove the smallest route by relocating its stops into other routes.
    Returns the updated route list and a success flag.
    """
    routeInfos = sorted(
        [(i, evaluateRoute(r, verbose=False)) for i, r in enumerate(allRoutes)],
        key=lambda x: (x[1]["total_cube"], x[1]["total_miles"])
    )

    for removeIdx, _ in routeInfos:
        routeToRemove = allRoutes[removeIdx]
        newRoutes = allRoutes.copy()
        success = True

        for stop in routeToRemove:
            targetIdx, newTargetRoute = bestRelocation(stop, newRoutes, removeIdx)
            if targetIdx is None:
                success = False
                break
            newRoutes[targetIdx] = newTargetRoute

        if success:
            newRoutes.pop(removeIdx)
            return newRoutes, True

    return allRoutes, False


def consolidateRoutes(allRoutes):
    """Repeatedly eliminate routes until no further reduction is possible."""
    improved = True
    while improved:
        allRoutes, improved = tryEliminateOneRoute(allRoutes)
    return allRoutes


def twoOptRoute(route):
    """2-opt intra-route improvement: reverse sub-sequences to reduce miles."""
    bestRoute = route[:]
    bestMiles = evaluateRoute(bestRoute, verbose=False)["total_miles"]
    improved = True

    while improved:
        improved = False
        for i in range(len(bestRoute) - 1):
            for j in range(i + 2, len(bestRoute)):
                trial = bestRoute[:i] + bestRoute[i:j + 1][::-1] + bestRoute[j + 1:]
                result = evaluateRoute(trial, verbose=False)

                if result["overall_feasible"] and result["total_miles"] < bestMiles:
                    bestRoute = trial
                    bestMiles = result["total_miles"]
                    improved = True
                    break
            if improved:
                break

    return bestRoute


def orOptRoute(route):
    """Or-opt: relocate chains of 1-3 stops to a cheaper position within the same route."""
    bestRoute = route[:]
    bestMiles = evaluateRoute(bestRoute, verbose=False)["total_miles"]
    improved = True

    while improved:
        improved = False
        for chainLen in [1, 2, 3]:
            for i in range(len(bestRoute) - chainLen + 1):
                chain = bestRoute[i:i + chainLen]
                remainder = bestRoute[:i] + bestRoute[i + chainLen:]

                for j in range(len(remainder) + 1):
                    if j == i:
                        continue
                    trial = remainder[:j] + chain + remainder[j:]
                    result = evaluateRoute(trial, verbose=False)

                    if result["overall_feasible"] and result["total_miles"] < bestMiles:
                        bestRoute = trial
                        bestMiles = result["total_miles"]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return bestRoute


def improveRoutes(allRoutes):
    """Apply 2-opt then or-opt to every route."""
    improved = []
    for route in allRoutes:
        if len(route) >= 4:
            route = twoOptRoute(route)
        if len(route) >= 2:
            route = orOptRoute(route)
        improved.append(route)
    return improved


def solveOneDay(dayOrders):
    """Run the full routing pipeline for one day."""
    if len(dayOrders) == 0:
        return []

    routes = clarkeWright(dayOrders)
    routes = consolidateRoutes(routes)
    routes = improveRoutes(routes)
    return routes


def solveSchedule(orders):
    """Solve all five weekdays and return routes keyed by day."""
    routesByDay = {}
    for day in DAYS:
        dayOrders = orders[orders["DayOfWeek"] == day].copy()
        routesByDay[day] = solveOneDay(dayOrders)
    return routesByDay


def getDayStats(routesByDay):
    """Return per-day miles and route counts; route count proxies daily driver/equipment demand."""
    stats = {}
    for day in DAYS:
        dayMiles = 0
        for route in routesByDay[day]:
            dayMiles += evaluateRoute(route, verbose=False)["total_miles"]
        stats[day] = {"routes": len(routesByDay[day]), "miles": dayMiles}
    return stats


def scheduleScore(dayStats, lambda_balance=25):
    """Total weekly miles plus a penalty for uneven daily route counts."""
    totalMiles = sum(dayStats[day]["miles"] for day in DAYS)
    routeCounts = [dayStats[day]["routes"] for day in DAYS]
    avgRoutes = sum(routeCounts) / len(routeCounts)
    imbalance = sum((count - avgRoutes) ** 2 for count in routeCounts)
    return totalMiles + lambda_balance * imbalance


def getVisitGroups(orders, store_col="TOZIP"):
    """Build moveable groups as (store, current day) to preserve weekly visit counts."""
    visitGroups = []
    grouped = orders.groupby([store_col, "DayOfWeek"], sort=False)

    for (store, day), group in grouped:
        visitGroups.append({
            "store":     int(store),
            "from_day":  day,
            "order_ids": group["ORDERID"].astype(int).tolist(),
        })

    return visitGroups


def buildSweepGroups(orders, store_col="TOZIP"):
    """
    Build movable store-day groups with angle and cube information for sweep assignment.
    """
    groups = []
    grouped = orders.groupby([store_col, "DayOfWeek"], sort=False)

    for (store, day), group in grouped:
        store = int(store)
        groups.append({
            "store": int(store),
            "from_day": day,
            "order_ids": group["ORDERID"].astype(int).tolist(),
            "total_cube": float(group["CUBE"].sum()),
            "order_count": int(len(group)),
            "angle": getAngleFromDepot(store),
        })

    return groups


def buildSweepAssignment(orders, store_col="TOZIP", start_idx=0, reverse=False):
    """
    Create an initial weekday schedule by angular sweep.
    """
    groups = buildSweepGroups(orders, store_col=store_col)
    if len(groups) == 0:
        return orders.copy()

    groups.sort(key=lambda g: g["angle"], reverse=reverse)

    start_idx = start_idx % len(groups)
    orderedGroups = groups[start_idx:] + groups[:start_idx]

    targetCubeByDay = (
        orders.groupby("DayOfWeek")["CUBE"]
        .sum()
        .reindex(DAYS, fill_value=0.0)
        .to_dict()
    )
    dayTargets = [float(targetCubeByDay[day]) for day in DAYS]

    updated = orders.copy()
    dayIdx = 0
    currentCube = 0.0

    for idx, group in enumerate(orderedGroups):
        remainingGroups = len(orderedGroups) - idx
        remainingDays = len(DAYS) - dayIdx
        groupCube = group["total_cube"]

        if dayIdx < len(DAYS) - 1 and currentCube > 0 and remainingGroups > remainingDays:
            target = dayTargets[dayIdx]
            cubeIfKeep = currentCube + groupCube

            gapWithoutGroup = abs(target - currentCube)
            gapWithGroup = abs(target - cubeIfKeep)

            if cubeIfKeep > target and gapWithoutGroup <= gapWithGroup:
                dayIdx += 1
                currentCube = 0.0

        assignedDay = DAYS[dayIdx]
        updated.loc[updated["ORDERID"].isin(group["order_ids"]), "DayOfWeek"] = assignedDay
        currentCube += groupCube

        remainingGroupsAfter = len(orderedGroups) - (idx + 1)
        remainingDaysAfter = len(DAYS) - (dayIdx + 1)

        if dayIdx < len(DAYS) - 1 and remainingGroupsAfter == remainingDaysAfter:
            dayIdx += 1
            currentCube = 0.0

    return updated


def buildBestSweepSchedule(orders, store_col="TOZIP", lambda_balance=25, n_starts=4):
    """
    Try several sweep rotations and keep the best one.
    """
    groups = buildSweepGroups(orders, store_col=store_col)
    if len(groups) == 0:
        routes = solveSchedule(orders.copy())
        stats = getDayStats(routes)
        info = {"start_idx": 0, "reverse": False, "candidates_tested": 0}
        return orders.copy(), routes, stats, info

    nCandidates = min(n_starts, len(groups))
    startIndices = sorted(set(int(i * len(groups) / nCandidates) for i in range(nCandidates)))

    bestOrders = None
    bestRoutes = None
    bestStats = None
    bestMiles = None
    bestScore = None
    bestInfo = None

    for reverse in [False, True]:
        for startIdx in startIndices:
            ROUTE_CACHE.clear()

            candidateOrders = buildSweepAssignment(
                orders,
                store_col=store_col,
                start_idx=startIdx,
                reverse=reverse,
            )
            candidateRoutes = solveSchedule(candidateOrders)
            candidateStats = getDayStats(candidateRoutes)
            candidateMiles = totalWeeklyMiles(candidateStats)
            candidateScore = scheduleScore(candidateStats, lambda_balance=lambda_balance)

            if bestOrders is None:
                bestOrders = candidateOrders
                bestRoutes = candidateRoutes
                bestStats = candidateStats
                bestMiles = candidateMiles
                bestScore = candidateScore
                bestInfo = {"start_idx": startIdx, "reverse": reverse}
            else:
                if (candidateMiles < bestMiles) or (
                    candidateMiles == bestMiles and candidateScore < bestScore
                ):
                    bestOrders = candidateOrders
                    bestRoutes = candidateRoutes
                    bestStats = candidateStats
                    bestMiles = candidateMiles
                    bestScore = candidateScore
                    bestInfo = {"start_idx": startIdx, "reverse": reverse}

    bestInfo["candidates_tested"] = len(startIndices) * 2
    return bestOrders, bestRoutes, bestStats, bestInfo


def moveVisitGroup(orders, order_ids, new_day):
    """Move one visit group (a set of orders) to a new weekday."""
    updated = orders.copy()
    updated.loc[updated["ORDERID"].isin(order_ids), "DayOfWeek"] = new_day
    return updated


def recomputeDayStatsForDay(routes):
    """Return route count and miles for one weekday."""
    dayMiles = 0
    for route in routes:
        dayMiles += evaluateRoute(route, verbose=False)["total_miles"]
    return {"routes": len(routes), "miles": dayMiles}


def tryMoveFast(bestOrders, bestRoutes, bestDayStats, group, newDay, lambda_balance=25):
    """Try moving one visit group to a new day; re-solves only the two affected days."""
    currentDay = group["from_day"]

    trialOrders = bestOrders.copy()
    trialOrders.loc[trialOrders["ORDERID"].isin(group["order_ids"]), "DayOfWeek"] = newDay

    trialRoutes = bestRoutes.copy()
    trialDayStats = bestDayStats.copy()

    for day in [currentDay, newDay]:
        dayOrders = trialOrders[trialOrders["DayOfWeek"] == day].copy()
        trialRoutes[day] = solveOneDay(dayOrders)
        trialDayStats[day] = recomputeDayStatsForDay(trialRoutes[day])

    trialScore = scheduleScore(trialDayStats, lambda_balance=lambda_balance)
    return trialOrders, trialRoutes, trialDayStats, trialScore


def totalWeeklyMiles(dayStats):
    return sum(dayStats[day]["miles"] for day in DAYS)


def relaxDeliveryDays(initialOrders, store_col="TOZIP", lambda_balance=25, max_passes=10, verbose=True):
    """
    Greedy local search that reduces weekly miles as the primary goal while secondarily improving day-to-day route balance.
    """
    bestOrders = initialOrders.copy()
    bestRoutes = solveSchedule(bestOrders)
    bestDayStats = getDayStats(bestRoutes)
    bestScore = scheduleScore(bestDayStats, lambda_balance=lambda_balance)
    bestMiles = totalWeeklyMiles(bestDayStats)

    acceptedMoves = []
    passNum = 0

    while passNum < max_passes:
        passNum += 1
        improvementFound = False

        visitGroups = getVisitGroups(bestOrders, store_col=store_col)
        routeCounts = {day: bestDayStats[day]["routes"] for day in DAYS}

        visitGroups.sort(
            key=lambda g: (-routeCounts[g["from_day"]], len(g["order_ids"]), g["store"])
        )

        for group in visitGroups:
            currentDay = group["from_day"]
            candidateDays = [day for day in DAYS if day != currentDay]
            candidateDays.sort(key=lambda d: (routeCounts[d], d))

            bestLocalChoice = None

            for newDay in candidateDays:
                trialOrders, trialRoutes, trialDayStats, trialScore = tryMoveFast(
                    bestOrders, bestRoutes, bestDayStats, group, newDay,
                    lambda_balance=lambda_balance
                )
                trialMiles = totalWeeklyMiles(trialDayStats)

                if trialMiles < bestMiles:
                    if bestLocalChoice is None:
                        bestLocalChoice = (trialOrders, trialRoutes, trialDayStats, trialScore, trialMiles, newDay)
                    else:
                        _, _, _, bestLocalScore, bestLocalMiles, _ = bestLocalChoice

                        if (trialMiles < bestLocalMiles) or (
                            trialMiles == bestLocalMiles and trialScore < bestLocalScore
                        ):
                            bestLocalChoice = (trialOrders, trialRoutes, trialDayStats, trialScore, trialMiles, newDay)

            if bestLocalChoice is not None:
                bestOrders, bestRoutes, bestDayStats, bestScore, bestMiles, chosenDay = bestLocalChoice

                acceptedMoves.append({
                    "store":     group["store"],
                    "order_ids": group["order_ids"],
                    "from_day":  currentDay,
                    "to_day":    chosenDay,
                })

                improvementFound = True

                if verbose:
                    print(
                        f"Accepted move {len(acceptedMoves)}: "
                        f"store {group['store']} | "
                        f"{currentDay} -> {chosenDay} | "
                        f"weekly miles = {bestMiles} | "
                        f"score = {round(bestScore, 2)}"
                    )
                break

        if not improvementFound:
            break

    return bestOrders, bestRoutes, bestDayStats, acceptedMoves, bestScore


def printScheduleSummary(title, routesByDay):
    """Print miles and route counts by weekday plus weekly total."""
    print(f"\n{title}")
    print("-" * 60)

    totalMiles = 0
    totalRoutes = 0

    for day in DAYS:
        dayMiles = 0
        for route in routesByDay[day]:
            dayMiles += evaluateRoute(route, verbose=False)["total_miles"]
        dayRoutes = len(routesByDay[day])
        print(f"{day}: routes={dayRoutes} | miles={dayMiles}")
        totalMiles += dayMiles
        totalRoutes += dayRoutes

    print("-" * 60)
    print(f"Weekly total routes: {totalRoutes}")
    print(f"Weekly total miles:  {totalMiles}")

    return totalMiles, totalRoutes


def printMoves(acceptedMoves):
    """Print the weekday reassignments that were accepted."""
    print("\nAccepted day changes")
    print("-" * 60)

    if not acceptedMoves:
        print("No improving weekday changes were found.")
        return

    for i, move in enumerate(acceptedMoves, start=1):
        print(
            f"{i}. Store {move['store']} | "
            f"{move['from_day']} -> {move['to_day']} | "
            f"orders {move['order_ids']}"
        )


def printDayReport(day, routes):
    """Print per-route details for one day; return (total_miles, total_orders)."""
    print(f"\n{day}")
    print("-" * 60)
    dayTotalMiles = 0
    dayTotalOrders = 0

    for i, route in enumerate(routes, start=1):
        r = evaluateRoute(route, verbose=False)
        orderCount = len(route)
        print(
            f"  Route {i}: {routeIds(route)} | "
            f"orders={orderCount} | "
            f"miles={r['total_miles']} | cube={r['total_cube']} | "
            f"drive={r['total_drive']}h | duty={r['total_duty']}h | "
            f"cap={r['capacity_feasible']} dot={r['dot_feasible']} "
            f"win={r['window_feasible']} | feasible={r['overall_feasible']}"
        )
        dayTotalMiles += r["total_miles"]
        dayTotalOrders += orderCount

    print(f"  {day} routes: {len(routes)} | orders: {dayTotalOrders} | total miles: {dayTotalMiles}")
    return dayTotalMiles, dayTotalOrders


def checkScheduleFeasibility(routesByDay):
    """Check every route in the weekly solution and report per-day and overall feasibility."""
    print("\nSchedule Feasibility Check")
    print("-" * 60)

    allFeasible = True

    for day in DAYS:
        print(f"\n{day}")
        dayFeasible = True

        for i, route in enumerate(routesByDay[day], start=1):
            r = evaluateRoute(route, verbose=False)
            print(
                f"  Route {i}: {routeIds(route)} | "
                f"cap={r['capacity_feasible']} | "
                f"window={r['window_feasible']} | "
                f"dot={r['dot_feasible']} | "
                f"overall={r['overall_feasible']}"
            )
            if not r["overall_feasible"]:
                dayFeasible = False
                allFeasible = False

        print(f"  {day} feasible: {dayFeasible}")

    print("\n" + "-" * 60)
    print(f"Weekly solution feasible: {allFeasible}")

    return allFeasible


def buildMap(routesByDay, zipCoords):
    """Build a Folium map for the final relaxed schedule."""
    depotCoord = zipCoords[DEPOT_ZIP]
    m = folium.Map(location=depotCoord, zoom_start=8, tiles="CartoDB positron")

    folium.Marker(
        location=depotCoord,
        tooltip=f"<b>Depot</b><br>ZIP {DEPOT_ZIP}",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)

    for day in DAYS:
        dayGroup = folium.FeatureGroup(name=day, show=True)
        routes = routesByDay[day]
        color = DAY_COLORS[day]

        for routeIdx, route in enumerate(routes, start=1):
            r = evaluateRoute(route, verbose=False)
            routeLabel = f"{day} · Route {routeIdx}"

            waypoints = [depotCoord]
            for stop in route:
                z = int(stop["TOZIP"])
                if z in zipCoords:
                    waypoints.append(zipCoords[z])
            waypoints.append(depotCoord)

            folium.PolyLine(
                locations=waypoints,
                color=color,
                weight=3,
                opacity=0.85,
                tooltip=(
                    f"{routeLabel} | {r['total_miles']} mi | "
                    f"{len(route)} orders | drive={r['total_drive']}h | duty={r['total_duty']}h"
                ),
            ).add_to(dayGroup)

            plugins.AntPath(
                locations=waypoints,
                color=color,
                weight=3,
                opacity=0.45,
                delay=1200,
                dash_array=[12, 24],
            ).add_to(dayGroup)

            for seq, stop in enumerate(route, start=1):
                z = int(stop["TOZIP"])
                if z not in zipCoords:
                    continue

                coord = zipCoords[z]
                oid = int(stop["ORDERID"])
                cube = int(stop["CUBE"])

                popupHtml = f"""
                <div style="font-family: monospace; font-size: 13px; min-width: 180px;">
                    <b>{routeLabel}</b><br>
                    Stop #{seq}<br>
                    Order ID: {oid}<br>
                    ZIP: {z}<br>
                    Cube: {cube} ft³<br>
                    Feasible: {r['overall_feasible']}
                </div>
                """

                folium.CircleMarker(
                    location=coord,
                    radius=7,
                    color="white",
                    weight=2,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.95,
                    tooltip=f"Stop {seq} · Order {oid} · ZIP {z}",
                    popup=folium.Popup(popupHtml, max_width=230),
                ).add_to(dayGroup)

                folium.Marker(
                    location=coord,
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="font-size:9px;font-weight:bold;'
                            f'color:white;text-align:center;line-height:14px;">'
                            f'{seq}</div>'
                        ),
                        icon_size=(14, 14),
                        icon_anchor=(7, 7),
                    ),
                ).add_to(dayGroup)

        dayGroup.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    legendHtml = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: white; border-radius: 8px; padding: 12px 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        font-family: 'Segoe UI', sans-serif; font-size: 13px;
    ">
        <b style="font-size:14px;">Days</b><br><br>
    """
    for day, color in DAY_COLORS.items():
        legendHtml += (
            f'<div style="display:flex;align-items:center;margin-bottom:5px;">'
            f'<div style="width:14px;height:14px;border-radius:50%;'
            f'background:{color};margin-right:8px;flex-shrink:0;"></div>'
            f'{day}</div>'
        )
    legendHtml += "</div>"
    m.get_root().html.add_child(folium.Element(legendHtml))

    return m


def mainRelaxedSchedule():
    """
    Hybrid question 3 approach:
    1. solve the original fixed schedule
    2. build a sweep-based initial weekday schedule
    3. refine that sweep schedule with local day-reassignment search
    4. check final feasibility
    5. build a map of the final relaxed routes
    """
    print("Solving baseline schedule...")
    baselineRoutes = solveSchedule(ORDERS)
    printScheduleSummary("Baseline Schedule", baselineRoutes)

    print("\nBuilding sweep-based initial weekday schedule...")
    sweepOrders, sweepRoutes, sweepDayStats, sweepInfo = buildBestSweepSchedule(
        ORDERS,
        store_col="TOZIP",
        lambda_balance=25,
        n_starts=4,
    )

    print(
        f"Chosen sweep: start_idx={sweepInfo['start_idx']} | "
        f"reverse={sweepInfo['reverse']} | "
        f"candidates_tested={sweepInfo['candidates_tested']}"
    )
    printScheduleSummary("Sweep Initial Schedule", sweepRoutes)

    print("\nRefining sweep schedule with local search...")
    relaxedOrders, relaxedRoutes, relaxedDayStats, acceptedMoves, relaxedScore = relaxDeliveryDays(
        sweepOrders,
        store_col="TOZIP",
        lambda_balance=25,
        max_passes=10,
        verbose=True,
    )

    printScheduleSummary("Final Relaxed Delivery-Day Schedule", relaxedRoutes)
    checkScheduleFeasibility(relaxedRoutes)
    printMoves(acceptedMoves)

    print("\nFinal reassigned weekday table")
    print("-" * 60)
    finalAssignments = (
        relaxedOrders[["ORDERID", "TOZIP", "DayOfWeek"]]
        .sort_values(["DayOfWeek", "TOZIP", "ORDERID"])
        .reset_index(drop=True)
    )
    print(finalAssignments.to_string(index=False))

    print("\nBuilding map...")
    m = buildMap(relaxedRoutes, ZIP_COORDS)
    m.save(OUTPUT_FILE)
    print(f"Map saved to: {OUTPUT_FILE}")


ORDERS, DIST_MATRIX = loadInputs()
ZIP_COORDS = loadZipCoords()

if __name__ == "__main__":
    mainRelaxedSchedule()