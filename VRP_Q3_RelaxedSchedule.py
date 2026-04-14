import pandas as pd
import warnings
from itertools import combinations

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

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

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
ROUTE_CACHE = {}


def load_inputs():
    """Read orders and distance matrix from Excel; coerce column types."""
    orders    = pd.read_excel("deliveries.xlsx", sheet_name="OrderTable")
    distances = pd.read_excel("distances.xlsx",  sheet_name="Sheet1")

    orders = orders[orders["ORDERID"] != 0].copy()
    orders["CUBE"]    = pd.to_numeric(orders["CUBE"],    errors="raise")
    orders["FROMZIP"] = pd.to_numeric(orders["FROMZIP"], errors="raise")
    orders["TOZIP"]   = pd.to_numeric(orders["TOZIP"],   errors="raise")
    orders["ORDERID"] = pd.to_numeric(orders["ORDERID"], errors="raise")

    distances = distances.rename(columns={"Unnamed: 0": "ZIP", "Unnamed: 1": "ZIPID"})
    distances = distances[distances["ZIP"] != "Zip"].copy()
    distances["ZIP"]   = pd.to_numeric(distances["ZIP"],   errors="raise")
    distances["ZIPID"] = pd.to_numeric(distances["ZIPID"], errors="raise")
    distances = distances.set_index("ZIP")

    distMatrix = distances.drop(columns=["ZIPID"]).copy()
    distMatrix.columns = pd.to_numeric(distMatrix.columns, errors="coerce")

    return orders, distMatrix


def get_distance(zip1, zip2):
    return DIST_MATRIX.loc[zip1, zip2]


def get_unload_time(cube):
    """Unload time in hours, enforcing a minimum floor of MIN_TIME minutes."""
    return max(MIN_TIME, UNLOAD_RATE * cube) / 60.0


def to_clock(hours):
    """Convert fractional hours to HH:MM string."""
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h += 1
        m = 0
    return f"{h:02d}:{m:02d}"


def route_ids(route):
    return [int(stop["ORDERID"]) for stop in route]

def route_key(route):
    return tuple(int(stop["ORDERID"]) for stop in route)

def evaluate_route(routeList, verbose=False):
    """
    Simulate the route and return feasibility flags and cost metrics.
    Checks three hard constraints: capacity, delivery windows, and DOT HOS.
    """
    if len(routeList) == 0:
        return {
            "total_miles": 0,
            "total_drive": 0.0,
            "total_unload": 0.0,
            "total_wait": 0.0,
            "total_duty": 0.0,
            "total_cube": 0,
            "return_time": 0.0,
            "capacity_feasible": True,
            "window_feasible": True,
            "dot_feasible": True,
            "overall_feasible": True,
        }

    cacheKey = None
    if not verbose:
        cacheKey = route_key(routeList)
        cached = ROUTE_CACHE.get(cacheKey)
        if cached is not None:
            return cached

    firstZip     = routeList[0]["TOZIP"]
    firstDrive   = get_distance(DEPOT_ZIP, firstZip) / DRIVING_SPEED
    dispatchTime = max(0.0, WINDOW_OPEN - firstDrive)

    currentZip  = DEPOT_ZIP
    currentTime = dispatchTime

    totalMiles     = 0
    totalDrive     = 0
    totalUnload    = 0
    totalWait      = 0
    totalCube      = 0
    windowFeasible = True

    if verbose:
        print("Dispatch:", to_clock(dispatchTime))

    for stop in routeList:
        stopZip      = stop["TOZIP"]
        cube         = stop["CUBE"]

        milesLeg     = get_distance(currentZip, stopZip)
        drive        = milesLeg / DRIVING_SPEED
        arrival      = currentTime + drive
        serviceStart = max(arrival, WINDOW_OPEN)
        wait         = max(0.0, WINDOW_OPEN - arrival)
        unload       = get_unload_time(cube)
        departure    = serviceStart + unload

        timeOfDay    = serviceStart % 24
        endOfService = departure % 24
        beforeClose  = (WINDOW_OPEN <= timeOfDay <= WINDOW_CLOSE) and (endOfService <= WINDOW_CLOSE)
        windowFeasible = windowFeasible and beforeClose

        if verbose:
            print(f"  Stop {int(stop['ORDERID'])}: arrive {to_clock(arrival)} | "
                  f"start {to_clock(serviceStart)} | depart {to_clock(departure)} | "
                  f"ok={beforeClose}")

        totalMiles  += milesLeg
        totalDrive  += drive
        totalWait   += wait
        totalUnload += unload
        totalCube   += cube

        currentTime = departure
        currentZip  = stopZip

    milesBack  = get_distance(currentZip, DEPOT_ZIP)
    driveBack  = milesBack / DRIVING_SPEED
    returnTime = currentTime + driveBack

    totalMiles += milesBack
    totalDrive += driveBack
    totalDuty   = totalDrive + totalUnload + totalWait

    capacityFeasible = totalCube  <= VAN_CAPACITY
    dotFeasible      = totalDrive <= MAX_DRIVING and totalDuty <= MAX_DUTY
    overallFeasible  = capacityFeasible and windowFeasible and dotFeasible

    results = {
        "total_miles":       int(totalMiles),
        "total_drive":       round(float(totalDrive),  3),
        "total_unload":      round(float(totalUnload), 3),
        "total_wait":        round(float(totalWait),   3),
        "total_duty":        round(float(totalDuty),   3),
        "total_cube":        int(totalCube),
        "return_time":       round(float(returnTime),  3),
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


def compute_savings(ordersDF):
    """
    Clarke-Wright savings: s(i,j) = d(depot,i) + d(depot,j) - d(i,j).
    Returns list of (savings, orderid_i, orderid_j) sorted descending.
    """
    orderList = ordersDF.to_dict("records")
    savings   = []

    for a, b in combinations(orderList, 2):
        zipA = a["TOZIP"]
        zipB = b["TOZIP"]
        s = (
            get_distance(DEPOT_ZIP, zipA)
            + get_distance(DEPOT_ZIP, zipB)
            - get_distance(zipA, zipB)
        )
        savings.append((s, int(a["ORDERID"]), int(b["ORDERID"])))

    savings.sort(key=lambda x: x[0], reverse=True)
    return savings


def clarke_wright(dayOrders):
    """
    Parallel savings construction. Each stop starts as its own route;
    pairs are merged in descending savings order when the merged route is feasible.
    Only valid endpoint adjacencies are considered (tail-to-head, with reversal allowed).
    """
    orderRecords = {int(row["ORDERID"]): row for _, row in dayOrders.iterrows()}

    routes  = {oid: [rec] for oid, rec in orderRecords.items()}
    routeOf = {oid: oid   for oid in orderRecords}
    headOf  = {oid: oid   for oid in orderRecords}
    tailOf  = {oid: oid   for oid in orderRecords}

    for s, oidI, oidJ in compute_savings(dayOrders):
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
            if evaluate_route(mergedRoute, verbose=False)["overall_feasible"]:
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


def best_relocation(stop, routes, skipRouteIdx):
    """Find the cheapest feasible insertion of stop across all routes except skipRouteIdx."""
    bestTargetIdx  = None
    bestNewRoute   = None
    bestExtraMiles = float("inf")

    for j, route in enumerate(routes):
        if j == skipRouteIdx:
            continue

        baseMiles = evaluate_route(route, verbose=False)["total_miles"]

        for pos in range(len(route) + 1):
            trialRoute = route[:pos] + [stop] + route[pos:]
            results    = evaluate_route(trialRoute, verbose=False)

            if results["overall_feasible"]:
                extraMiles = results["total_miles"] - baseMiles
                if extraMiles < bestExtraMiles:
                    bestTargetIdx  = j
                    bestNewRoute   = trialRoute
                    bestExtraMiles = extraMiles

    return bestTargetIdx, bestNewRoute


def try_eliminate_one_route(allRoutes):
    """
    Try to remove the smallest route by relocating its stops into other routes.
    Returns the updated route list and a success flag.
    """
    routeInfos = sorted(
        [(i, evaluate_route(r, verbose=False)) for i, r in enumerate(allRoutes)],
        key=lambda x: (x[1]["total_cube"], x[1]["total_miles"])
    )

    for removeIdx, _ in routeInfos:
        routeToRemove = allRoutes[removeIdx]
        newRoutes     = allRoutes.copy()
        success       = True

        for stop in routeToRemove:
            targetIdx, newTargetRoute = best_relocation(stop, newRoutes, removeIdx)
            if targetIdx is None:
                success = False
                break
            newRoutes[targetIdx] = newTargetRoute

        if success:
            newRoutes.pop(removeIdx)
            return newRoutes, True

    return allRoutes, False


def consolidate_routes(allRoutes):
    """Repeatedly eliminate routes until no further reduction is possible."""
    improved = True
    while improved:
        allRoutes, improved = try_eliminate_one_route(allRoutes)
    return allRoutes


def two_opt_route(route):
    """2-opt intra-route improvement: reverse sub-sequences to reduce miles."""
    bestRoute = route[:]
    bestMiles = evaluate_route(bestRoute, verbose=False)["total_miles"]
    improved  = True

    while improved:
        improved = False
        for i in range(len(bestRoute) - 1):
            for j in range(i + 2, len(bestRoute)):
                trial  = bestRoute[:i] + bestRoute[i:j + 1][::-1] + bestRoute[j + 1:]
                result = evaluate_route(trial, verbose=False)

                if result["overall_feasible"] and result["total_miles"] < bestMiles:
                    bestRoute = trial
                    bestMiles = result["total_miles"]
                    improved  = True
                    break
            if improved:
                break

    return bestRoute


def or_opt_route(route):
    """Or-opt: relocate chains of 1-3 stops to a cheaper position within the same route."""
    bestRoute = route[:]
    bestMiles = evaluate_route(bestRoute, verbose=False)["total_miles"]
    improved  = True

    while improved:
        improved = False
        for chainLen in [1, 2, 3]:
            for i in range(len(bestRoute) - chainLen + 1):
                chain     = bestRoute[i:i + chainLen]
                remainder = bestRoute[:i] + bestRoute[i + chainLen:]

                for j in range(len(remainder) + 1):
                    if j == i:
                        continue
                    trial  = remainder[:j] + chain + remainder[j:]
                    result = evaluate_route(trial, verbose=False)

                    if result["overall_feasible"] and result["total_miles"] < bestMiles:
                        bestRoute = trial
                        bestMiles = result["total_miles"]
                        improved  = True
                        break
                if improved:
                    break
            if improved:
                break

    return bestRoute


def improve_routes(allRoutes):
    """Apply 2-opt then or-opt to every route."""
    improved = []
    for route in allRoutes:
        if len(route) >= 4:
            route = two_opt_route(route)
        if len(route) >= 2:
            route = or_opt_route(route)
        improved.append(route)
    return improved

def solve_one_day(dayOrders):
    """Run your existing construction + improvement pipeline for one day."""
    if len(dayOrders) == 0:
        return []

    routes = clarke_wright(dayOrders)
    routes = consolidate_routes(routes)
    routes = improve_routes(routes)
    return routes


def solve_schedule(orders):
    """Solve all five weekdays using the existing routing code."""
    routesByDay = {}

    for day in DAYS:
        dayOrders = orders[orders["DayOfWeek"] == day].copy()
        routesByDay[day] = solve_one_day(dayOrders)

    return routesByDay


def get_day_stats(routesByDay):
    """
    Return per-day miles and route counts.
    Route count is used as a proxy for daily drivers/equipment needed.
    """
    stats = {}

    for day in DAYS:
        dayMiles = 0
        for route in routesByDay[day]:
            dayMiles += evaluate_route(route, verbose=False)["total_miles"]

        stats[day] = {
            "routes": len(routesByDay[day]),
            "miles": dayMiles
        }

    return stats


def schedule_score(dayStats, lambda_balance=25):
    """
    Objective for question 3:
      total weekly miles + penalty for uneven daily route counts.

    Increase lambda_balance if you want more emphasis on stable day-by-day
    resource requirements.
    """
    totalMiles = sum(dayStats[day]["miles"] for day in DAYS)

    routeCounts = [dayStats[day]["routes"] for day in DAYS]
    avgRoutes   = sum(routeCounts) / len(routeCounts)

    imbalancePenalty = sum((count - avgRoutes) ** 2 for count in routeCounts)

    return totalMiles + lambda_balance * imbalancePenalty


def get_visit_groups(orders, store_col="TOZIP"):
    """
    Build moveable groups as (store, current day).

    This is better than moving an entire store across all days, because some stores
    may have multiple weekly visits. Grouping by (store, day) preserves the number
    of weekly visits while still allowing the assigned weekday to change.
    """
    visitGroups = []

    grouped = orders.groupby([store_col, "DayOfWeek"], sort=False)

    for (store, day), group in grouped:
        visitGroups.append({
            "store": int(store),
            "from_day": day,
            "order_ids": group["ORDERID"].astype(int).tolist()
        })

    return visitGroups


def move_visit_group(orders, order_ids, new_day):
    """Move one visit group (a set of orders) to a new weekday."""
    updated = orders.copy()
    updated.loc[updated["ORDERID"].isin(order_ids), "DayOfWeek"] = new_day
    return updated

def recompute_day_stats_for_day(routes):
    """Return route count and miles for one weekday."""
    dayMiles = 0
    for route in routes:
        dayMiles += evaluate_route(route, verbose=False)["total_miles"]

    return {
        "routes": len(routes),
        "miles": dayMiles
    }


def try_move_fast(bestOrders, bestRoutes, bestDayStats, group, newDay, lambda_balance=25):
    """
    Try moving one visit group to a new day.
    Only re-solve the two affected days.
    """
    currentDay = group["from_day"]

    trialOrders = bestOrders.copy()
    trialOrders.loc[trialOrders["ORDERID"].isin(group["order_ids"]), "DayOfWeek"] = newDay

    trialRoutes   = bestRoutes.copy()
    trialDayStats = bestDayStats.copy()

    affectedDays = [currentDay, newDay]

    for day in affectedDays:
        dayOrders = trialOrders[trialOrders["DayOfWeek"] == day].copy()
        trialRoutes[day] = solve_one_day(dayOrders)
        trialDayStats[day] = recompute_day_stats_for_day(trialRoutes[day])

    trialScore = schedule_score(trialDayStats, lambda_balance=lambda_balance)

    return trialOrders, trialRoutes, trialDayStats, trialScore

def total_weekly_miles(dayStats):
    return sum(dayStats[day]["miles"] for day in DAYS)

def relax_delivery_days(initialOrders, store_col="TOZIP", lambda_balance=25, max_passes=10, verbose=True):
    """
    Faster greedy local search for question 3.

    Primary goal: reduce weekly miles.
    Secondary goal: improve day-to-day balance.
    """
    bestOrders   = initialOrders.copy()
    bestRoutes   = solve_schedule(bestOrders)
    bestDayStats = get_day_stats(bestRoutes)
    bestScore    = schedule_score(bestDayStats, lambda_balance=lambda_balance)
    bestMiles    = total_weekly_miles(bestDayStats)

    acceptedMoves = []
    passNum = 0

    while passNum < max_passes:
        passNum += 1
        improvementFound = False

        visitGroups = get_visit_groups(bestOrders, store_col=store_col)

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
                trialOrders, trialRoutes, trialDayStats, trialScore = try_move_fast(
                    bestOrders,
                    bestRoutes,
                    bestDayStats,
                    group,
                    newDay,
                    lambda_balance=lambda_balance
                )

                trialMiles = total_weekly_miles(trialDayStats)

                # Only consider moves that reduce miles
                if trialMiles < bestMiles:
                    if bestLocalChoice is None:
                        bestLocalChoice = (trialOrders, trialRoutes, trialDayStats, trialScore, trialMiles, newDay)
                    else:
                        _, _, _, bestLocalScore, bestLocalMiles, _ = bestLocalChoice

                        # Prefer lower miles first; use score as tie-breaker
                        if (trialMiles < bestLocalMiles) or (trialMiles == bestLocalMiles and trialScore < bestLocalScore):
                            bestLocalChoice = (trialOrders, trialRoutes, trialDayStats, trialScore, trialMiles, newDay)

            if bestLocalChoice is not None:
                bestOrders, bestRoutes, bestDayStats, bestScore, bestMiles, chosenDay = bestLocalChoice

                acceptedMoves.append({
                    "store": group["store"],
                    "order_ids": group["order_ids"],
                    "from_day": currentDay,
                    "to_day": chosenDay
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

def print_schedule_summary(title, routesByDay):
    """Print miles and route counts by weekday plus weekly total."""
    print(f"\n{title}")
    print("-" * 60)

    totalMiles  = 0
    totalRoutes = 0

    for day in DAYS:
        dayMiles = 0
        for route in routesByDay[day]:
            dayMiles += evaluate_route(route, verbose=False)["total_miles"]

        dayRoutes = len(routesByDay[day])

        print(
            f"{day}: routes={dayRoutes} | miles={dayMiles}"
        )

        totalMiles  += dayMiles
        totalRoutes += dayRoutes

    print("-" * 60)
    print(f"Weekly total routes: {totalRoutes}")
    print(f"Weekly total miles:  {totalMiles}")

    return totalMiles, totalRoutes


def print_moves(acceptedMoves):
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

def print_day_report(day, routes):
    """Print per-route details for one day; return (total_miles, total_orders)."""
    print(f"\n{day}")
    print("-" * 60)
    dayTotalMiles  = 0
    dayTotalOrders = 0

    for i, route in enumerate(routes, start=1):
        r          = evaluate_route(route, verbose=False)
        orderCount = len(route)
        print(
            f"  Route {i}: {route_ids(route)} | "
            f"orders={orderCount} | "
            f"miles={r['total_miles']} | cube={r['total_cube']} | "
            f"drive={r['total_drive']}h | duty={r['total_duty']}h | "
            f"cap={r['capacity_feasible']} dot={r['dot_feasible']} "
            f"win={r['window_feasible']} | feasible={r['overall_feasible']}"
        )
        dayTotalMiles  += r["total_miles"]
        dayTotalOrders += orderCount

    print(f"  {day} routes: {len(routes)} | orders: {dayTotalOrders} | total miles: {dayTotalMiles}")
    return dayTotalMiles, dayTotalOrders

def check_schedule_feasibility(routesByDay):
    print("\nSchedule Feasibility Check")
    print("-" * 60)

    all_feasible = True

    for day in DAYS:
        print(f"\n{day}")
        day_feasible = True

        for i, route in enumerate(routesByDay[day], start=1):
            r = evaluate_route(route, verbose=False)

            print(
                f"  Route {i}: {route_ids(route)} | "
                f"cap={r['capacity_feasible']} | "
                f"window={r['window_feasible']} | "
                f"dot={r['dot_feasible']} | "
                f"overall={r['overall_feasible']}"
            )

            if not r["overall_feasible"]:
                day_feasible = False
                all_feasible = False

        print(f"  {day} feasible: {day_feasible}")

    print("\n" + "-" * 60)
    print(f"Weekly solution feasible: {all_feasible}")

    return all_feasible

def main_relaxed_schedule():
    """
    Question 3 analysis:
    Compare the current fixed weekday schedule to a relaxed schedule.
    """
    print("Solving baseline schedule...")
    baselineRoutes = solve_schedule(ORDERS)
    print_schedule_summary("Baseline Schedule", baselineRoutes)

    print("\nSearching for improved weekday assignments...")
    relaxedOrders, relaxedRoutes, relaxedDayStats, acceptedMoves, relaxedScore = relax_delivery_days(
        ORDERS,
        store_col="TOZIP",      # using destination zip as store identifier
        lambda_balance=25,     # increase if you want more even weekday route counts
        max_passes=10,
        verbose=True
    )

    print_schedule_summary("Relaxed Delivery-Day Schedule", relaxedRoutes)
    check_schedule_feasibility(relaxedRoutes)
    print_moves(acceptedMoves)

    print("\nFinal reassigned weekday table")
    print("-" * 60)
    finalAssignments = (
        relaxedOrders[["ORDERID", "TOZIP", "DayOfWeek"]]
        .sort_values(["DayOfWeek", "TOZIP", "ORDERID"])
        .reset_index(drop=True)
    )
    print(finalAssignments.to_string(index=False))



ORDERS, DIST_MATRIX = load_inputs()

if __name__ == "__main__":
    main_relaxed_schedule()
