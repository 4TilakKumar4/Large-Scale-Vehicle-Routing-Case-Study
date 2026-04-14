"""
vrp_solvers/overnightSolver.py — Overnight DOT break routing logic and OvernightSolver class.

Wraps any base solver (CW, ALNS, etc.) and applies overnight pairings on top of its
day-cab solution. Deliveries still occur within 08:00-18:00 store windows; the overnight
break is a logistics event between the last delivery on day 1 and the first on day 2.

Exports: OvernightSolver, evaluateOvernightRoute, applyOvernightImprovements.
"""

import time

from vrp_solvers.base import (
    BREAK_TIME,
    DAYS,
    DEPOT_ZIP,
    DRIVING_SPEED,
    MAX_DRIVING,
    MAX_DUTY,
    VAN_CAPACITY,
    WINDOW_CLOSE,
    WINDOW_OPEN,
    evaluateRoute,
    getDistance,
    getUnloadTime,
    toClock,
)

ADJACENT_DAY_PAIRS = [
    ("Mon", "Tue"),
    ("Tue", "Wed"),
    ("Wed", "Thu"),
    ("Thu", "Fri"),
]


def serveRouteSegment(routeList, startZip, startTime, driveUsed=0, dutyUsed=0, verbose=False):
    """
    Simulate driving a sequence of stops from a given position with accumulated HOS.
    Used by evaluateOvernightRoute to evaluate each day's segment independently,
    preserving drive/duty continuity across the overnight break.
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

        milesLeg     = getDistance(currentZip, stopZip)
        drive        = milesLeg / DRIVING_SPEED
        arrival      = currentTime + drive
        serviceStart = max(arrival, WINDOW_OPEN)
        wait         = max(0.0, WINDOW_OPEN - arrival)
        unload       = getUnloadTime(cube)
        departure    = serviceStart + unload

        # Check service both starts AND completes within the delivery window
        timeOfDay   = serviceStart % 24
        endOfDay    = departure % 24
        beforeClose = (WINDOW_OPEN <= timeOfDay <= WINDOW_CLOSE) and (endOfDay <= WINDOW_CLOSE)
        windowFeasible = windowFeasible and beforeClose

        if verbose:
            print(f"  Stop {int(stop['ORDERID'])}: arrive {toClock(arrival)} | "
                  f"start {toClock(serviceStart)} | depart {toClock(departure)} | "
                  f"ok={beforeClose}")

        totalMiles  += milesLeg
        totalDrive  += drive
        totalWait   += wait
        totalUnload += unload
        totalCube   += cube

        currentTime = departure
        currentZip  = stopZip

    totalDuty = dutyUsed + totalUnload + totalWait + (totalDrive - driveUsed)

    return {
        "end_zip":         currentZip,
        "end_time":        currentTime,
        "total_miles":     totalMiles,
        "total_drive":     totalDrive,
        "total_unload":    totalUnload,
        "total_wait":      totalWait,
        "total_duty":      totalDuty,
        "total_cube":      totalCube,
        "window_feasible": windowFeasible,
    }


def _latestBreakTransition(lastZip, firstNextZip, finishTime, driveUsed, dutyUsed):
    """
    Compute the latest-possible DOT break placement between day-1 and day-2 stops.
    Driver travels as far as legally permitted before stopping, takes the 10-hour break,
    then resumes. Two cases: driver reaches the next stop before HOS runs out (break on
    arrival), or HOS runs out en route (break mid-journey).
    """
    remainingDrive   = MAX_DRIVING - driveUsed
    remainingDuty    = MAX_DUTY    - dutyUsed
    legalTravelHours = min(remainingDrive, remainingDuty)

    if legalTravelHours < 0:
        return {"feasible": False, "reason": "day 1 already exceeds drive/duty limit"}

    travelNeededHours = getDistance(lastZip, firstNextZip) / DRIVING_SPEED
    day2OpenTime      = 24.0 + WINDOW_OPEN

    if travelNeededHours <= legalTravelHours:
        arrivalBDay1   = finishTime + travelNeededHours
        breakStartTime = arrivalBDay1
        breakEndTime   = breakStartTime + BREAK_TIME
        arrivalBDay2   = breakEndTime
        day2WaitAtB    = max(0.0, day2OpenTime - arrivalBDay2)
        serviceStartB  = arrivalBDay2 + day2WaitAtB

        return {
            "feasible":                       True,
            "case":                           "reach_B_before_break",
            "day1_added_miles":               getDistance(lastZip, firstNextZip),
            "day1_added_drive":               travelNeededHours,
            "day1_added_wait":                0.0,
            "day1_final_drive":               driveUsed + travelNeededHours,
            "day1_final_duty":                dutyUsed  + travelNeededHours,
            "break_start_time":               breakStartTime,
            "break_end_time":                 breakEndTime,
            "day2_arrival_B":                 arrivalBDay2,
            "service_start_B":                serviceStartB,
            "day2_drive_used_before_service": 0.0,
            "day2_duty_used_before_service":  day2WaitAtB,
            "day2_added_miles":               0.0,
            "day2_added_wait":                day2WaitAtB,
        }

    breakStartTime = finishTime + legalTravelHours
    breakEndTime   = breakStartTime + BREAK_TIME

    day1AddedDrive = legalTravelHours
    day1AddedMiles = legalTravelHours * DRIVING_SPEED

    remainingToB  = travelNeededHours - legalTravelHours
    arrivalBDay2  = breakEndTime + remainingToB
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
        "day2_drive_used_before_service": remainingToB,
        "day2_duty_used_before_service":  remainingToB + day2WaitAtB,
        "day2_added_miles":               remainingToB * DRIVING_SPEED,
        "day2_added_wait":                day2WaitAtB,
    }


def evaluateOvernightRoute(day1Route, day2Route, verbose=False):
    """
    Evaluate a two-day route spanning one overnight DOT break.
    Day-1 stops are served first; the driver then travels toward the first day-2 stop
    until HOS runs out, takes a 10-hour break, and resumes. All deliveries must fall
    within the 08:00-18:00 store window on their respective day.
    """
    if not day1Route or not day2Route:
        return {"overall_feasible": False,
                "reason": "both day1Route and day2Route must be non-empty"}

    firstZipDay1  = day1Route[0]["TOZIP"]
    firstDriveDay1 = getDistance(DEPOT_ZIP, firstZipDay1) / DRIVING_SPEED
    dispatchTime   = max(0.0, WINDOW_OPEN - firstDriveDay1)

    if verbose:
        print("Dispatch day 1:", toClock(dispatchTime))

    seg1 = serveRouteSegment(
        day1Route,
        startZip=DEPOT_ZIP,
        startTime=dispatchTime,
        driveUsed=0,
        dutyUsed=0,
        verbose=verbose,
    )

    firstZipDay2 = day2Route[0]["TOZIP"]
    trans = _latestBreakTransition(
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
        print(f"  Case:         {trans['case']}")
        print(f"  Break starts: {toClock(trans['break_start_time'])}")
        print(f"  Break ends:   {toClock(trans['break_end_time'])}")
        print(f"  Day-2 service starts: {toClock(trans['service_start_B'])}")

    seg2 = serveRouteSegment(
        day2Route,
        startZip=firstZipDay2,
        startTime=trans["service_start_B"],
        driveUsed=trans["day2_drive_used_before_service"],
        dutyUsed=trans["day2_duty_used_before_service"],
        verbose=verbose,
    )

    milesBack  = getDistance(seg2["end_zip"], DEPOT_ZIP)
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
        print("\nReturn to depot:", toClock(returnTime))
        print("\nTotals:", results)

    return results


def _compareOvernightPair(route1, route2):
    """Return savings dict if combining route1 + route2 overnight reduces total miles."""
    separateMiles = (
        evaluateRoute(route1)["total_miles"]
        + evaluateRoute(route2)["total_miles"]
    )
    overnight = evaluateOvernightRoute(route1, route2)

    if overnight["overall_feasible"] and overnight["total_miles"] < separateMiles:
        return {
            "improves":          True,
            "separate_miles":    separateMiles,
            "overnight_miles":   overnight["total_miles"],
            "savings":           separateMiles - overnight["total_miles"],
            "overnight_results": overnight,
        }

    return {"improves": False, "separate_miles": separateMiles}


def findAllOvernightCandidates(routesDay1, routesDay2):
    """Return all improving overnight pairs sorted by savings descending."""
    candidates = []

    for i, r1 in enumerate(routesDay1):
        for j, r2 in enumerate(routesDay2):
            comp = _compareOvernightPair(r1, r2)
            if comp["improves"]:
                candidates.append({
                    "route1_idx":        i,
                    "route2_idx":        j,
                    "savings":           comp["savings"],
                    "separate_miles":    comp["separate_miles"],
                    "overnight_miles":   comp["overnight_miles"],
                    "overnight_results": comp["overnight_results"],
                })

    candidates.sort(key=lambda x: x["savings"], reverse=True)
    return candidates


def applyOvernightImprovements(routesByDay):
    """
    Greedily apply non-conflicting overnight pairings in descending savings order.
    Each route can appear in at most one pairing.
    Returns (overnightRoutes list, usedRoutes dict of day → set of consumed route indices).
    """
    overnightRoutes = []
    usedRoutes      = {day: set() for day in routesByDay}

    for d1, d2 in ADJACENT_DAY_PAIRS:
        candidates = findAllOvernightCandidates(
            routesByDay.get(d1, []), routesByDay.get(d2, [])
        )

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


class OvernightSolver:
    """
    Wraps any base solver and applies overnight pairings on top of its weekly solution.
    Unlike the per-day solvers, solve() takes the full orders DataFrame because overnight
    pairing requires looking at adjacent days simultaneously.
    Call solve(); retrieve results with getStats(). getConvergence() returns None.
    """

    def __init__(self, baseSolver):
        self.baseSolver       = baseSolver
        self._stats           = None
        self._routesByDay     = None
        self._overnightRoutes = None
        self._usedRoutes      = None

    def solve(self, orders):
        """Run base solver on all days then apply overnight pairings; return (routesByDay, overnightRoutes, usedRoutes)."""
        t0 = time.time()

        routesByDay = {}
        for day in DAYS:
            dayOrders        = orders[orders["DayOfWeek"] == day].copy()
            routesByDay[day] = self.baseSolver.solve(dayOrders)

        overnightRoutes, usedRoutes = applyOvernightImprovements(routesByDay)

        self._routesByDay     = routesByDay
        self._overnightRoutes = overnightRoutes
        self._usedRoutes      = usedRoutes
        self._stats           = self._collectStats(
            routesByDay, overnightRoutes, usedRoutes, time.time() - t0
        )

        return routesByDay, overnightRoutes, usedRoutes

    def getStats(self):
        return self._stats

    def getConvergence(self):
        return None

    def getWeightHistory(self):
        return None

    def getOvernightRoutes(self):
        return self._overnightRoutes

    def _collectStats(self, routesByDay, overnightRoutes, usedRoutes, elapsed):
        totalMiles  = 0
        totalRoutes = 0
        allFeasible = True

        for day in DAYS:
            for idx, route in enumerate(routesByDay.get(day, [])):
                if idx not in usedRoutes[day]:
                    r = evaluateRoute(route)
                    totalMiles  += r["total_miles"]
                    totalRoutes += 1
                    if not r["overall_feasible"]:
                        allFeasible = False

        for ovn in overnightRoutes:
            totalMiles  += ovn["results"]["total_miles"]
            totalRoutes += 1
            if not ovn["results"]["overall_feasible"]:
                allFeasible = False

        return {
            "miles":            totalMiles,
            "routes":           totalRoutes,
            "feasible":         allFeasible,
            "runtime_s":        round(elapsed, 2),
            "overnight_pairs":  len(overnightRoutes),
        }