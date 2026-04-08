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

DAYS               = ["Mon", "Tue", "Wed", "Thu", "Fri"]
ADJACENT_DAY_PAIRS = [("Mon", "Tue"), ("Tue", "Wed"), ("Wed", "Thu"), ("Thu", "Fri")]


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

    # Fix column dtype so integer ZIP lookups work correctly
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


def serve_route_segment(routeList, startZip, startTime, driveUsed=0, dutyUsed=0, verbose=False):
    """
    Simulate driving a sequence of stops from a given starting position and
    accumulated drive/duty time. Returns end state and totals for the segment.
    Used by both evaluate_route and evaluate_overnight_route.
    """
    currentZip  = startZip
    currentTime = startTime

    totalMiles     = 0
    totalDrive     = driveUsed
    totalUnload    = 0
    totalWait      = 0
    totalCube      = 0
    windowFeasible = True

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

        # Check service both starts AND completes within window
        timeOfDay  = serviceStart % 24
        endOfDay   = departure % 24
        beforeClose = (WINDOW_OPEN <= timeOfDay <= WINDOW_CLOSE) and (endOfDay <= WINDOW_CLOSE)
        windowFeasible = windowFeasible and beforeClose

        if verbose:
            print("Stop:", int(stop["ORDERID"]))
            print(" Arrive:", to_clock(arrival))
            print(" Start service:", to_clock(serviceStart))
            print(" Depart:", to_clock(departure))
            print(" Before close:", beforeClose)

        totalMiles  += milesLeg
        totalDrive  += drive
        totalWait   += wait
        totalUnload += unload
        totalCube   += cube

        currentTime = departure
        currentZip  = stopZip

    totalDuty = dutyUsed + totalUnload + totalWait + (totalDrive - driveUsed)

    return {
        "end_zip":        currentZip,
        "end_time":       currentTime,
        "total_miles":    totalMiles,
        "total_drive":    totalDrive,
        "total_unload":   totalUnload,
        "total_wait":     totalWait,
        "total_duty":     totalDuty,
        "total_cube":     totalCube,
        "window_feasible": windowFeasible,
    }


def evaluate_route(routeList, verbose=False):
    """
    Simulate the route and return feasibility flags and cost metrics.
    Checks three hard constraints: capacity, delivery windows, and DOT HOS.
    """
    firstZip     = routeList[0]["TOZIP"]
    firstDrive   = get_distance(DEPOT_ZIP, firstZip) / DRIVING_SPEED
    dispatchTime = WINDOW_OPEN - firstDrive

    # Guard: dispatch cannot be before previous midnight
    if dispatchTime < 0:
        dispatchTime = 0.0

    if verbose:
        print("Dispatch:", to_clock(dispatchTime))

    seg = serve_route_segment(
        routeList,
        startZip=DEPOT_ZIP,
        startTime=dispatchTime,
        driveUsed=0,
        dutyUsed=0,
        verbose=verbose,
    )

    milesBack  = get_distance(seg["end_zip"], DEPOT_ZIP)
    driveBack  = milesBack / DRIVING_SPEED
    returnTime = seg["end_time"] + driveBack

    totalMiles  = seg["total_miles"] + milesBack
    totalDrive  = seg["total_drive"] + driveBack
    totalUnload = seg["total_unload"]
    totalWait   = seg["total_wait"]
    totalDuty   = totalDrive + totalUnload + totalWait
    totalCube   = seg["total_cube"]

    capacityFeasible = totalCube  <= VAN_CAPACITY
    dotFeasible      = totalDrive <= MAX_DRIVING and totalDuty <= MAX_DUTY
    overallFeasible  = capacityFeasible and seg["window_feasible"] and dotFeasible

    results = {
        "total_miles":       int(totalMiles),
        "total_drive":       round(float(totalDrive),  3),
        "total_unload":      round(float(totalUnload), 3),
        "total_wait":        round(float(totalWait),   3),
        "total_duty":        round(float(totalDuty),   3),
        "total_cube":        int(totalCube),
        "return_time":       round(float(returnTime),  3),
        "capacity_feasible": bool(capacityFeasible),
        "window_feasible":   bool(seg["window_feasible"]),
        "dot_feasible":      bool(dotFeasible),
        "overall_feasible":  bool(overallFeasible),
    }

    if verbose:
        print("\nReturn to depot:", to_clock(returnTime))
        print("\nTotals")
        print(results)

    return results


def latest_break_transition(lastZip, firstNextZip, finishTime, driveUsed, dutyUsed):
    """
    Compute the latest-possible DOT break transition between day-1 and day-2 stops.
    The driver travels as far as legally allowed before stopping, then resumes after
    the 10-hour break. Returns miles, drive, wait, and timing for both day segments.
    """
    remainingDrive     = MAX_DRIVING - driveUsed
    remainingDuty      = MAX_DUTY    - dutyUsed
    legalTravelHours   = min(remainingDrive, remainingDuty)

    if legalTravelHours < 0:
        return {"feasible": False, "reason": "day 1 already exceeds drive/duty limit"}

    travelNeededHours = get_distance(lastZip, firstNextZip) / DRIVING_SPEED
    day2OpenTime      = 24 + WINDOW_OPEN

    if travelNeededHours <= legalTravelHours:
        # Driver reaches next stop before hitting HOS limit; break on arrival
        arrivalBDay1   = finishTime + travelNeededHours
        breakStartTime = arrivalBDay1
        waitAtBDay1    = 0.0
        breakEndTime   = breakStartTime + BREAK_TIME

        arrivalBDay2   = breakEndTime
        day2WaitAtB    = max(0.0, day2OpenTime - arrivalBDay2)
        serviceStartB  = arrivalBDay2 + day2WaitAtB

        return {
            "feasible":                       True,
            "case":                           "reach_B_before_break",
            "day1_added_miles":               get_distance(lastZip, firstNextZip),
            "day1_added_drive":               travelNeededHours,
            "day1_added_wait":                waitAtBDay1,
            "day1_final_drive":               driveUsed + travelNeededHours,
            "day1_final_duty":                dutyUsed  + travelNeededHours + waitAtBDay1,
            "break_start_time":               breakStartTime,
            "break_end_time":                 breakEndTime,
            "day2_arrival_B":                 arrivalBDay2,
            "service_start_B":                serviceStartB,
            "day2_drive_used_before_service": 0.0,
            "day2_duty_used_before_service":  day2WaitAtB,
            "day2_added_miles":               0.0,
            "day2_added_wait":                day2WaitAtB,
        }

    # Driver hits HOS limit en route; break wherever legal travel runs out
    breakStartTime = finishTime + legalTravelHours
    breakEndTime   = breakStartTime + BREAK_TIME

    day1AddedDrive = legalTravelHours
    day1AddedMiles = legalTravelHours * DRIVING_SPEED

    remainingHoursToBAfterBreak = travelNeededHours - legalTravelHours
    arrivalBDay2  = breakEndTime + remainingHoursToBAfterBreak
    day2WaitAtB   = max(0.0, day2OpenTime - arrivalBDay2)
    serviceStartB = arrivalBDay2 + day2WaitAtB

    return {
        "feasible":                       True,
        "case":                           "break_en_route",
        "day1_added_miles":               day1AddedMiles,
        "day1_added_drive":               day1AddedDrive,
        "day1_added_wait":                0.0,
        "day1_final_drive":               driveUsed + day1AddedDrive,
        "day1_final_duty":                dutyUsed  + day1AddedDrive,
        "break_start_time":               breakStartTime,
        "break_end_time":                 breakEndTime,
        "day2_arrival_B":                 arrivalBDay2,
        "service_start_B":                serviceStartB,
        "day2_drive_used_before_service": remainingHoursToBAfterBreak,
        "day2_duty_used_before_service":  remainingHoursToBAfterBreak + day2WaitAtB,
        "day2_added_miles":               remainingHoursToBAfterBreak * DRIVING_SPEED,
        "day2_added_wait":                day2WaitAtB,
    }


def evaluate_overnight_route(day1Route, day2Route, verbose=False):
    """
    Evaluate a two-day route spanning one overnight DOT break.
    Runs day-1 stops, computes the latest-break transition, then runs day-2 stops.
    Returns combined feasibility flags and metrics for both days.
    """
    if not day1Route or not day2Route:
        return {"overall_feasible": False, "reason": "Both day1Route and day2Route must be nonempty"}

    firstZipDay1  = day1Route[0]["TOZIP"]
    firstDriveDay1 = get_distance(DEPOT_ZIP, firstZipDay1) / DRIVING_SPEED
    dispatchTime  = max(0.0, WINDOW_OPEN - firstDriveDay1)

    if verbose:
        print("Dispatch day 1:", to_clock(dispatchTime))

    seg1 = serve_route_segment(
        day1Route,
        startZip=DEPOT_ZIP,
        startTime=dispatchTime,
        driveUsed=0,
        dutyUsed=0,
        verbose=verbose,
    )

    firstZipDay2 = day2Route[0]["TOZIP"]
    trans = latest_break_transition(
        lastZip=seg1["end_zip"],
        firstNextZip=firstZipDay2,
        finishTime=seg1["end_time"],
        driveUsed=seg1["total_drive"],
        dutyUsed=seg1["total_duty"],
    )

    if not trans["feasible"]:
        return {"overall_feasible": False, "reason": trans["reason"]}

    if verbose:
        print("\nOvernight transition")
        print(" Case:", trans["case"])
        print(" Break starts:", to_clock(trans["break_start_time"]))
        print(" Break ends:",   to_clock(trans["break_end_time"]))
        print(" Service at first day-2 stop starts:", to_clock(trans["service_start_B"]))

    seg2 = serve_route_segment(
        day2Route,
        startZip=firstZipDay2,
        startTime=trans["service_start_B"],
        driveUsed=trans["day2_drive_used_before_service"],
        dutyUsed=trans["day2_duty_used_before_service"],
        verbose=verbose,
    )

    milesBack  = get_distance(seg2["end_zip"], DEPOT_ZIP)
    driveBack  = milesBack / DRIVING_SPEED
    returnTime = seg2["end_time"] + driveBack

    totalMiles = (
        seg1["total_miles"]
        + trans["day1_added_miles"]
        + trans["day2_added_miles"]
        + seg2["total_miles"]
        + milesBack
    )

    totalWait = (
        seg1["total_wait"]
        + trans["day1_added_wait"]
        + trans["day2_added_wait"]
        + seg2["total_wait"]
    )

    totalUnload = seg1["total_unload"] + seg2["total_unload"]
    totalCube   = seg1["total_cube"]   + seg2["total_cube"]

    day1Drive = trans["day1_final_drive"]
    day1Duty  = trans["day1_final_duty"]
    day2Drive = seg2["total_drive"] + driveBack
    day2Duty  = seg2["total_duty"]  + driveBack

    capacityFeasible = totalCube <= VAN_CAPACITY
    windowFeasible   = seg1["window_feasible"] and seg2["window_feasible"]
    day1DotFeasible  = day1Drive <= MAX_DRIVING and day1Duty <= MAX_DUTY
    day2DotFeasible  = day2Drive <= MAX_DRIVING and day2Duty <= MAX_DUTY
    dotFeasible      = day1DotFeasible and day2DotFeasible
    overallFeasible  = capacityFeasible and windowFeasible and dotFeasible

    results = {
        "total_miles":       int(totalMiles),
        "total_unload":      round(float(totalUnload), 3),
        "total_wait":        round(float(totalWait),   3),
        "total_cube":        int(totalCube),
        "day1_drive":        round(float(day1Drive),   3),
        "day1_duty":         round(float(day1Duty),    3),
        "day2_drive":        round(float(day2Drive),   3),
        "day2_duty":         round(float(day2Duty),    3),
        "break_start_time":  round(float(trans["break_start_time"]), 3),
        "break_end_time":    round(float(trans["break_end_time"]),   3),
        "return_time":       round(float(returnTime),  3),
        "capacity_feasible": bool(capacityFeasible),
        "window_feasible":   bool(windowFeasible),
        "day1_dot_feasible": bool(day1DotFeasible),
        "day2_dot_feasible": bool(day2DotFeasible),
        "dot_feasible":      bool(dotFeasible),
        "overall_feasible":  bool(overallFeasible),
    }

    if verbose:
        print("\nReturn to depot:", to_clock(returnTime))
        print("\nTotals")
        print(results)

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

                if dropRid in headOf:
                    del headOf[dropRid]
                if dropRid in tailOf:
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
    routeInfos = []
    for i, route in enumerate(allRoutes):
        r = evaluate_route(route, verbose=False)
        routeInfos.append((i, r["total_cube"], r["total_miles"]))

    routeInfos.sort(key=lambda x: (x[1], x[2]))

    for removeIdx, _, _ in routeInfos:
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
    bestRoute   = route[:]
    bestResults = evaluate_route(bestRoute, verbose=False)
    improved    = True

    while improved:
        improved = False
        for i in range(len(bestRoute) - 1):
            for j in range(i + 2, len(bestRoute)):
                trialRoute   = bestRoute[:i] + bestRoute[i:j + 1][::-1] + bestRoute[j + 1:]
                trialResults = evaluate_route(trialRoute, verbose=False)

                if trialResults["overall_feasible"] and trialResults["total_miles"] < bestResults["total_miles"]:
                    bestRoute   = trialRoute
                    bestResults = trialResults
                    improved    = True
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


def compare_overnight_pair(route1, route2):
    """Return savings dict if combining route1 and route2 overnight reduces total miles."""
    separateMiles = (
        evaluate_route(route1, verbose=False)["total_miles"]
        + evaluate_route(route2, verbose=False)["total_miles"]
    )
    overnight = evaluate_overnight_route(route1, route2, verbose=False)

    if overnight["overall_feasible"] and overnight["total_miles"] < separateMiles:
        return {
            "improves":         True,
            "separate_miles":   separateMiles,
            "overnight_miles":  overnight["total_miles"],
            "savings":          separateMiles - overnight["total_miles"],
            "overnight_results": overnight,
        }

    return {"improves": False, "separate_miles": separateMiles}


def find_all_overnight_candidates(routesDay1, routesDay2):
    """Return all improving overnight pairs sorted by savings descending."""
    candidates = []

    for i, r1 in enumerate(routesDay1):
        for j, r2 in enumerate(routesDay2):
            comp = compare_overnight_pair(r1, r2)
            if comp["improves"]:
                candidates.append({
                    "route1_idx":       i,
                    "route2_idx":       j,
                    "savings":          comp["savings"],
                    "separate_miles":   comp["separate_miles"],
                    "overnight_miles":  comp["overnight_miles"],
                    "overnight_results": comp["overnight_results"],
                })

    candidates.sort(key=lambda x: x["savings"], reverse=True)
    return candidates


def apply_overnight_improvements(routesByDay):
    """
    Greedily apply non-conflicting overnight pairings in descending savings order.
    Each route can appear in at most one overnight pairing.
    """
    overnightRoutes = []
    usedRoutes      = {day: set() for day in routesByDay}

    for d1, d2 in ADJACENT_DAY_PAIRS:
        candidates = find_all_overnight_candidates(routesByDay[d1], routesByDay[d2])

        for cand in candidates:
            i = cand["route1_idx"]
            j = cand["route2_idx"]

            if i in usedRoutes[d1] or j in usedRoutes[d2]:
                continue  # already consumed by a better pairing

            usedRoutes[d1].add(i)
            usedRoutes[d2].add(j)

            overnightRoutes.append({
                "day1":       d1,
                "day2":       d2,
                "route1_idx": i,
                "route2_idx": j,
                "route1":     routesByDay[d1][i],
                "route2":     routesByDay[d2][j],
                "results":    cand["overnight_results"],
                "savings":    cand["savings"],
            })

    return overnightRoutes, usedRoutes


def route_summary(route):
    """Return a compact metrics dict for one route."""
    r = evaluate_route(route, verbose=False)
    return {
        "orders":   route_ids(route),
        "miles":    r["total_miles"],
        "cube":     r["total_cube"],
        "duty":     r["total_duty"],
        "feasible": r["overall_feasible"],
    }


def print_day_report(day, routes):
    """Print per-route details for one day; return total miles."""
    print(f"\n{day}")
    print("-" * 60)
    dayTotalMiles = 0

    for i, route in enumerate(routes, start=1):
        s = route_summary(route)
        print(
            f"  Route {i}: {s['orders']} | "
            f"miles={s['miles']} | cube={s['cube']} | "
            f"duty={s['duty']} | feasible={s['feasible']}"
        )
        dayTotalMiles += s["miles"]

    print(f"  {day} route count: {len(routes)}")
    print(f"  {day} total miles: {dayTotalMiles}")
    return dayTotalMiles


def print_overnight_candidates(routesByDay):
    """Print all improving overnight pairs for each adjacent day pair."""
    for d1, d2 in ADJACENT_DAY_PAIRS:
        candidates = find_all_overnight_candidates(routesByDay[d1], routesByDay[d2])
        print(f"\nOvernight candidates: {d1} -> {d2}")
        if not candidates:
            print("  No improving overnight pairs found.")
        else:
            for cand in candidates:
                print(
                    f"  {d1} Route {cand['route1_idx'] + 1} + "
                    f"{d2} Route {cand['route2_idx'] + 1} | "
                    f"separate={cand['separate_miles']} | "
                    f"overnight={cand['overnight_miles']} | "
                    f"savings={cand['savings']}"
                )


def print_applied_overnights(overnightRoutes):
    """Print each overnight pairing that was applied."""
    print("\nApplied Overnight Routes")
    print("-" * 60)
    if not overnightRoutes:
        print("  No overnight routes applied.")
        return

    for k, ovn in enumerate(overnightRoutes, start=1):
        print(
            f"  Overnight {k}: {ovn['day1']} Route {ovn['route1_idx'] + 1} "
            f"+ {ovn['day2']} Route {ovn['route2_idx'] + 1} | "
            f"day1_orders={route_ids(ovn['route1'])} | "
            f"day2_orders={route_ids(ovn['route2'])} | "
            f"savings={ovn['savings']} | "
            f"miles={ovn['results']['total_miles']} | "
            f"feasible={ovn['results']['overall_feasible']}"
        )


def main():
    """Build routes, run improvement passes, report, then apply overnight pairings."""
    routesByDay = {}

    for day in DAYS:
        print(f"Building routes for {day}...")
        dayOrders = ORDERS[ORDERS["DayOfWeek"] == day].copy()
        routes    = clarke_wright(dayOrders)
        routes    = consolidate_routes(routes)
        routes    = improve_routes(routes)
        routesByDay[day] = routes

    weeklyTotalMiles  = 0
    weeklyTotalRoutes = 0

    for day in DAYS:
        weeklyTotalMiles  += print_day_report(day, routesByDay[day])
        weeklyTotalRoutes += len(routesByDay[day])

    print("\nWeekly Summary (before overnight)")
    print("-" * 60)
    print("Total weekly routes:", weeklyTotalRoutes)
    print("Total weekly miles:",  weeklyTotalMiles)

    print_overnight_candidates(routesByDay)

    overnightRoutes, usedRoutes = apply_overnight_improvements(routesByDay)
    print_applied_overnights(overnightRoutes)

    finalTotalMiles  = 0
    finalTotalRoutes = 0

    for day in DAYS:
        for idx, route in enumerate(routesByDay[day]):
            if idx not in usedRoutes[day]:
                finalTotalMiles  += evaluate_route(route, verbose=False)["total_miles"]
                finalTotalRoutes += 1

    for ovn in overnightRoutes:
        finalTotalMiles  += ovn["results"]["total_miles"]
        finalTotalRoutes += 1

    print("\nFinal Summary with Overnight")
    print("-" * 60)
    print("Total routes:", finalTotalRoutes)
    print("Total miles:",  finalTotalMiles)


ORDERS, DIST_MATRIX = load_inputs()

if __name__ == "__main__":
    main()
