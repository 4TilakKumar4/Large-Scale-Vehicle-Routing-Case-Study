"""
vrp_solvers/base.py — Shared constants, data I/O, route evaluation, and local search
utilities used by every solver in the package.

Input  : data/orders_clean.csv, data/distance_matrix.csv (written by VRP_DataAnalysis.py)
Exports: constants, loadInputs, evaluateRoute, twoOptRoute, orOptRoute,
         applyLocalSearch, consolidateRoutes, and small helpers.
"""

import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

VAN_CAPACITY  = 3200
ST_CAPACITY   = 1400
ST_UNLOAD_RATE = 0.043  # min / cubic foot — lift-gate slower than van
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

# Module-level globals populated by loadInputs()
ORDERS      = None
DIST_MATRIX = None


def loadInputs():
    """Read cleaned orders and distance matrix from data/."""
    global ORDERS, DIST_MATRIX

    ordersPath = os.path.join(DATA_DIR, "orders_clean.csv")
    distPath   = os.path.join(DATA_DIR, "distance_matrix.csv")

    # Catch missing data/ files with a clear message — the most common cause
    # is running a solver before VRP_DataAnalysis.py has been executed
    try:
        orders = pd.read_csv(ordersPath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"orders_clean.csv not found at {ordersPath}\n"
            "Run VRP_DataAnalysis.py first to generate the data/ directory."
        )

    try:
        distMatrix = pd.read_csv(distPath, index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"distance_matrix.csv not found at {distPath}\n"
            "Run VRP_DataAnalysis.py first to generate the data/ directory."
        )

    orders["CUBE"]    = pd.to_numeric(orders["CUBE"],    errors="raise")
    orders["FROMZIP"] = pd.to_numeric(orders["FROMZIP"], errors="raise")
    orders["TOZIP"]   = pd.to_numeric(orders["TOZIP"],   errors="raise")
    orders["ORDERID"] = pd.to_numeric(orders["ORDERID"], errors="raise")

    distMatrix.index   = pd.to_numeric(distMatrix.index,   errors="coerce")
    distMatrix.columns = pd.to_numeric(distMatrix.columns, errors="coerce")

    # Guard against a distance matrix that parsed but is effectively empty
    if distMatrix.empty:
        raise ValueError(
            "distance_matrix.csv loaded but produced an empty DataFrame. "
            "Re-run VRP_DataAnalysis.py to regenerate it."
        )

    ORDERS      = orders
    DIST_MATRIX = distMatrix

    return orders, distMatrix


def getDistance(zip1, zip2):
    # Guard against calling before loadInputs() has been run
    if DIST_MATRIX is None:
        raise RuntimeError(
            "DIST_MATRIX is not loaded. Call loadInputs() before using getDistance()."
        )

    if zip1 not in DIST_MATRIX.index:
        raise KeyError(f"ZIP {zip1} not found in distance matrix index.")
    if zip2 not in DIST_MATRIX.columns:
        raise KeyError(f"ZIP {zip2} not found in distance matrix columns.")

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


def evaluateRoute(routeList, verbose=False):
    """
    Simulate the route and return feasibility flags and cost metrics.
    Checks three hard constraints: capacity, delivery windows, and DOT HOS.
    """
    if not routeList:
        raise ValueError(
            "evaluateRoute() received an empty route list. "
            "Routes must contain at least one stop."
        )

    firstZip     = routeList[0]["TOZIP"]
    firstDrive   = getDistance(DEPOT_ZIP, firstZip) / DRIVING_SPEED
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
        print("Dispatch:", toClock(dispatchTime))

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

        timeOfDay    = serviceStart % 24
        endOfService = departure % 24
        beforeClose  = (WINDOW_OPEN <= timeOfDay <= WINDOW_CLOSE) and (endOfService <= WINDOW_CLOSE)
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

    milesBack  = getDistance(currentZip, DEPOT_ZIP)
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

    if verbose:
        print("Return to depot:", toClock(returnTime))
        print("Totals:", results)

    return results


def twoOptRoute(route):
    """2-opt intra-route improvement: reverse sub-sequences to reduce miles."""
    bestRoute = route[:]
    bestMiles = evaluateRoute(bestRoute)["total_miles"]
    improved  = True

    while improved:
        improved = False
        for i in range(len(bestRoute) - 1):
            for j in range(i + 2, len(bestRoute)):
                trial  = bestRoute[:i] + bestRoute[i:j + 1][::-1] + bestRoute[j + 1:]
                result = evaluateRoute(trial)

                if result["overall_feasible"] and result["total_miles"] < bestMiles:
                    bestRoute = trial
                    bestMiles = result["total_miles"]
                    improved  = True
                    break
            if improved:
                break

    return bestRoute


def orOptRoute(route):
    """Or-opt: relocate chains of 1-3 stops to a cheaper position within the same route."""
    bestRoute = route[:]
    bestMiles = evaluateRoute(bestRoute)["total_miles"]
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
                    result = evaluateRoute(trial)

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


def applyLocalSearch(routes):
    """Apply 2-opt then or-opt to every route; used as a shared polishing step."""
    if not routes:
        return routes

    polished = []
    for route in routes:
        if len(route) >= 4:
            route = twoOptRoute(route)
        if len(route) >= 2:
            route = orOptRoute(route)
        polished.append(route)
    return polished


def _bestRelocation(stop, routes, skipIdx):
    """Find the cheapest feasible insertion of stop across all routes except skipIdx."""
    bestTargetIdx  = None
    bestNewRoute   = None
    bestExtraMiles = float("inf")

    for j, route in enumerate(routes):
        if j == skipIdx:
            continue

        baseMiles = evaluateRoute(route)["total_miles"]

        for pos in range(len(route) + 1):
            trial  = route[:pos] + [stop] + route[pos:]
            result = evaluateRoute(trial)

            if result["overall_feasible"]:
                extraMiles = result["total_miles"] - baseMiles
                if extraMiles < bestExtraMiles:
                    bestTargetIdx  = j
                    bestNewRoute   = trial
                    bestExtraMiles = extraMiles

    return bestTargetIdx, bestNewRoute


def _tryEliminateOneRoute(allRoutes):
    """
    Try to remove the smallest route by relocating all its stops into other routes.
    Returns the updated list and a success flag.
    """
    routeInfos = sorted(
        [(i, evaluateRoute(r)) for i, r in enumerate(allRoutes)],
        key=lambda x: (x[1]["total_cube"], x[1]["total_miles"])
    )

    for removeIdx, _ in routeInfos:
        routeToRemove = allRoutes[removeIdx]
        newRoutes     = allRoutes.copy()
        success       = True

        for stop in routeToRemove:
            targetIdx, newTargetRoute = _bestRelocation(stop, newRoutes, removeIdx)
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
    if not allRoutes:
        return allRoutes

    improved = True
    while improved:
        allRoutes, improved = _tryEliminateOneRoute(allRoutes)
    return allRoutes


def detailedRouteTrace(route, day, routeNum, locs=None):
    """
    Re-simulate a route and return per-stop timing rows matching Table 3 format.
    Each row has: day, route_number, stop_sequence, order_id, location,
    arrival_time, departure_time, delivery_volume_cuft.
    Depot appears as the first and last row with no arrival/volume on dispatch
    and no departure/volume on return.
    locs is an optional DataFrame with ZIP→CITY mapping; if None the ZIP is used.
    """
    if not route:
        return []

    zipToCity = {}
    if locs is not None:
        zipToCity = dict(zip(locs["ZIP"].astype(int), locs["CITY"]))

    def cityName(z):
        return zipToCity.get(int(z), str(int(z)))

    firstZip     = route[0]["TOZIP"]
    firstDrive   = getDistance(DEPOT_ZIP, firstZip) / DRIVING_SPEED
    dispatchTime = max(0.0, WINDOW_OPEN - firstDrive)

    rows = []

    # Opening depot row
    rows.append({
        "day":                    day,
        "route_number":           routeNum,
        "stop_sequence":          0,
        "order_id":               0,
        "location":               "DC (Wilmington)",
        "arrival_time":           None,
        "departure_time":         toClock(dispatchTime),
        "delivery_volume_cuft":   None,
    })

    currentZip  = DEPOT_ZIP
    currentTime = dispatchTime
    seq         = 1

    for stop in route:
        stopZip      = stop["TOZIP"]
        cube         = stop["CUBE"]

        milesLeg     = getDistance(currentZip, stopZip)
        drive        = milesLeg / DRIVING_SPEED
        arrival      = currentTime + drive
        serviceStart = max(arrival, WINDOW_OPEN)
        unload       = getUnloadTime(cube)
        departure    = serviceStart + unload

        rows.append({
            "day":                    day,
            "route_number":           routeNum,
            "stop_sequence":          seq,
            "order_id":               int(stop["ORDERID"]),
            "location":               cityName(stopZip),
            "arrival_time":           toClock(arrival),
            "departure_time":         toClock(departure),
            "delivery_volume_cuft":   int(cube),
        })

        currentTime = departure
        currentZip  = stopZip
        seq        += 1

    # Closing depot row
    milesBack  = getDistance(currentZip, DEPOT_ZIP)
    driveBack  = milesBack / DRIVING_SPEED
    returnTime = currentTime + driveBack

    rows.append({
        "day":                    day,
        "route_number":           routeNum,
        "stop_sequence":          seq,
        "order_id":               0,
        "location":               "DC (Wilmington)",
        "arrival_time":           toClock(returnTime),
        "departure_time":         None,
        "delivery_volume_cuft":   None,
    })

    return rows


def getUnloadTimeMixed(cube, vehicleType):
    """Unload time in hours for the given vehicle type."""
    rate = ST_UNLOAD_RATE if vehicleType == "st" else UNLOAD_RATE
    return max(MIN_TIME, rate * cube) / 60.0


def evaluateMixedRoute(routeList, vehicleType, verbose=False):
    """
    Evaluate a route for a specific vehicle type — van or st.
    Identical to evaluateRoute() but uses vehicle-specific capacity and unload rate.
    vehicleType must be "van" or "st".
    """
    if not routeList:
        raise ValueError("evaluateMixedRoute() received an empty route list.")

    capacity = VAN_CAPACITY if vehicleType == "van" else ST_CAPACITY

    firstZip     = routeList[0]["TOZIP"]
    firstDrive   = getDistance(DEPOT_ZIP, firstZip) / DRIVING_SPEED
    dispatchTime = max(0.0, WINDOW_OPEN - firstDrive)

    currentZip  = DEPOT_ZIP
    currentTime = dispatchTime

    totalMiles     = 0
    totalDrive     = 0
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
        unload       = getUnloadTimeMixed(cube, vehicleType)
        departure    = serviceStart + unload

        timeOfDay    = serviceStart % 24
        endOfService = departure % 24
        beforeClose  = (WINDOW_OPEN <= timeOfDay <= WINDOW_CLOSE) and (endOfService <= WINDOW_CLOSE)
        windowFeasible = windowFeasible and beforeClose

        totalMiles  += milesLeg
        totalDrive  += drive
        totalWait   += wait
        totalUnload += unload
        totalCube   += cube

        currentTime = departure
        currentZip  = stopZip

    milesBack  = getDistance(currentZip, DEPOT_ZIP)
    driveBack  = milesBack / DRIVING_SPEED
    returnTime = currentTime + driveBack

    totalMiles += milesBack
    totalDrive += driveBack
    totalDuty   = totalDrive + totalUnload + totalWait

    capacityFeasible = totalCube  <= capacity
    dotFeasible      = totalDrive <= MAX_DRIVING and totalDuty <= MAX_DUTY
    overallFeasible  = capacityFeasible and windowFeasible and dotFeasible

    return {
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
