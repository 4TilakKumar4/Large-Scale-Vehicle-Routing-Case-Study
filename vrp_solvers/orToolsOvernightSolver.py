"""
vrp_solvers/orToolsOvernightSolver.py — OR-Tools two-day overnight routing (Q3-A).

Models each adjacent day-pair (Mon-Tue, Tue-Wed, Wed-Thu, Thu-Fri) as a single
RoutingModel with a two-day time horizon. A mandatory 10-hour DOT break interval
is injected per vehicle so OR-Tools places the rest period optimally rather than
the planner placing it manually.

Constraint stack vs Q1 (orToolsSolver.py):
  Capacity  — unchanged, cubic-foot load cap per van
  Time      — two-day horizon; day-1 nodes pinned to [480,1080],
              day-2 nodes pinned to [1920,2520] (=day-1 + 1440 min)
  Break     — mandatory FixedDurationIntervalVar of 600 min per vehicle,
              earliest start = MAX_DRIVING (660 min), replaces DriveTime dim
  DutyTime  — cumulative drive+service, capped at MAX_DUTY per day-segment

This is Option A: existing day assignments are respected; only the routing
across a two-day window and the break placement are optimised.

Exports: ORToolsOvernightSolver
"""

import math
import time

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from vrp_solvers.base import (
    BREAK_TIME,
    DEPOT_ZIP,
    DRIVING_SPEED,
    MAX_DUTY,
    MAX_DRIVING,
    MIN_TIME,
    UNLOAD_RATE,
    VAN_CAPACITY,
    WINDOW_CLOSE,
    WINDOW_OPEN,
    evaluateRoute,
    getDistance,
)

_MINS_PER_HOUR  = 60
_DAY_MIN        = 24 * _MINS_PER_HOUR             # 1440 — one full day in minutes

_OPEN_MIN       = int(WINDOW_OPEN  * _MINS_PER_HOUR)   # 480
_CLOSE_MIN      = int(WINDOW_CLOSE * _MINS_PER_HOUR)   # 1080
_MAX_DRIVE_MIN  = int(MAX_DRIVING  * _MINS_PER_HOUR)   # 660
_MAX_DUTY_MIN   = int(MAX_DUTY     * _MINS_PER_HOUR)   # 840
_BREAK_MIN      = int(BREAK_TIME   * _MINS_PER_HOUR)   # 600

# Day-2 windows are shifted by exactly one day
_OPEN_MIN_D2    = _OPEN_MIN  + _DAY_MIN                # 1920
_CLOSE_MIN_D2   = _CLOSE_MIN + _DAY_MIN                # 2520

# Two-day horizon: latest possible return = day-2 close + full duty budget
_HORIZON        = _CLOSE_MIN_D2 + _MAX_DUTY_MIN        # 3360


def _toMins(miles):
    """Road miles → integer travel minutes at DRIVING_SPEED mph."""
    return int(round(miles * _MINS_PER_HOUR / DRIVING_SPEED))


def _serviceMins(cube):
    """Unload time in integer minutes, respecting the MIN_TIME floor."""
    return int(round(max(MIN_TIME, UNLOAD_RATE * cube)))


class ORToolsOvernightSolver:
    """
    Two-day CVRP with mandatory 10-hour DOT break modelled as a break interval.
    Call solvePair() for each adjacent day-pair; retrieve results with getStats().
    Day-1 and day-2 routes are returned separately and keyed by day name.
    """

    def __init__(self, timeLimitSec=60, extraVehicles=2):
        self.timeLimitSec  = timeLimitSec
        self.extraVehicles = extraVehicles
        self._stats        = {}

    def solvePair(self, day1Name, day1Orders, day2Name, day2Orders):
        """
        Solve one adjacent pair and return {day1Name: routes, day2Name: routes}.
        Records keyed by day name are lists of order-record lists (same format
        as every other solver in the package).
        """
        if day1Orders.empty and day2Orders.empty:
            self._stats[f"{day1Name}-{day2Name}"] = {
                "miles": 0, "routes": 0, "feasible": True, "runtime_s": 0.0
            }
            return {day1Name: [], day2Name: []}

        t0 = time.time()

        day1Records = day1Orders.to_dict("records")
        day2Records = day2Orders.to_dict("records")

        routesByDay = self._buildRoutes(day1Name, day1Records, day2Name, day2Records)

        elapsed = time.time() - t0
        self._stats[f"{day1Name}-{day2Name}"] = self._collectStats(
            routesByDay[day1Name] + routesByDay[day2Name], elapsed
        )
        return routesByDay

    def getStats(self):
        """Return stats dict keyed by 'Day1-Day2' pair strings."""
        return self._stats

    # ------------------------------------------------------------------

    def _buildRoutes(self, day1Name, day1Records, day2Name, day2Records):
        nDay1    = len(day1Records)
        nDay2    = len(day2Records)
        allRecs  = day1Records + day2Records
        n        = len(allRecs)

        # Node layout: 0=depot, 1..nDay1=day-1 stops, nDay1+1..n=day-2 stops
        nodeZip  = [DEPOT_ZIP] + [r["TOZIP"] for r in allRecs]
        nodeCube = [0]         + [r["CUBE"]  for r in allRecs]

        # Day tag per node (0=depot, 1=day1, 2=day2) — used for window pinning
        nodeDay = [0] + [1] * nDay1 + [2] * nDay2

        travelMin = [
            [
                0 if i == j else _toMins(getDistance(nodeZip[i], nodeZip[j]))
                for j in range(n + 1)
            ]
            for i in range(n + 1)
        ]
        serviceMin = [0] + [_serviceMins(r["CUBE"]) for r in allRecs]

        totalCube       = sum(r["CUBE"] for r in allRecs)
        totalServiceMin = sum(_serviceMins(r["CUBE"]) for r in allRecs)

        capBound  = math.ceil(totalCube       / VAN_CAPACITY)
        hosBound  = math.ceil(totalServiceMin / _MAX_DUTY_MIN)
        stopBound = math.ceil(n               / 8)
        numVehicles = max(capBound, hosBound, stopBound) + self.extraVehicles

        manager = pywrapcp.RoutingIndexManager(n + 1, numVehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        # ── arc-cost: travel minutes ────────────────────────────────────
        def distCb(fromIdx, toIdx):
            return travelMin[manager.IndexToNode(fromIdx)][manager.IndexToNode(toIdx)]

        distCbIdx = routing.RegisterTransitCallback(distCb)
        routing.SetArcCostEvaluatorOfAllVehicles(distCbIdx)

        # ── capacity dimension ──────────────────────────────────────────
        def demandCb(fromIdx):
            return int(nodeCube[manager.IndexToNode(fromIdx)])

        demandCbIdx = routing.RegisterUnaryTransitCallback(demandCb)
        routing.AddDimensionWithVehicleCapacity(
            demandCbIdx,
            0,
            [VAN_CAPACITY] * numVehicles,
            True,
            "Capacity",
        )

        # ── time dimension: two-day horizon ─────────────────────────────
        # Transit = travel time + service time at from-node.
        # Slack allows waiting at a node until its window opens.
        # The horizon covers both days plus a full duty budget of headroom.
        def timeCb(fromIdx, toIdx):
            fromNode = manager.IndexToNode(fromIdx)
            toNode   = manager.IndexToNode(toIdx)
            return travelMin[fromNode][toNode] + serviceMin[fromNode]

        timeCbIdx = routing.RegisterTransitCallback(timeCb)
        routing.AddDimension(
            timeCbIdx,
            _OPEN_MIN_D2,   # max slack: a vehicle can wait at most until day-2 opens
            _HORIZON,
            False,          # departure time is a free variable per vehicle
            "Time",
        )
        timeDim = routing.GetDimensionOrDie("Time")

        # Pin each store node to its day's delivery window
        for i in range(1, n + 1):
            idx = manager.NodeToIndex(i)
            if nodeDay[i] == 1:
                timeDim.CumulVar(idx).SetRange(_OPEN_MIN,    _CLOSE_MIN)
            else:
                timeDim.CumulVar(idx).SetRange(_OPEN_MIN_D2, _CLOSE_MIN_D2)

        # ── duty-time dimension (drive + service per shift) ─────────────
        # Reuses timeCbIdx. Capped at MAX_DUTY so each day-segment stays
        # within the 14-hour on-duty limit even across the two-day span.
        routing.AddDimension(
            timeCbIdx,
            0,
            _MAX_DUTY_MIN,
            True,
            "DutyTime",
        )

        # ── mandatory 10-hour break per vehicle ─────────────────────────
        # FixedDurationIntervalVar: must start no earlier than MAX_DRIVING
        # minutes into the shift, last exactly BREAK_TIME minutes, and
        # complete before the horizon. OR-Tools places the break optimally.
        solver = routing.solver()
        for v in range(numVehicles):
            breakInterval = solver.FixedDurationIntervalVar(
                _MAX_DRIVE_MIN,   # earliest start: after 11h of driving
                _HORIZON,         # latest start: anywhere before horizon
                _BREAK_MIN,       # fixed duration: 600 min (10h)
                False,            # not optional — every vehicle must rest
                f"break_v{v}",
            )
            routing.SetBreakIntervalsOfVehicle(
                [breakInterval],
                v,
                [],               # no pre-travel times needed here
            )

        # ── search parameters ───────────────────────────────────────────
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        params.time_limit.seconds = self.timeLimitSec

        solution = routing.SolveWithParameters(params)

        if solution is None:
            print(f"  ORToolsOvernightSolver: no solution for {day1Name}-{day2Name} "
                  f"— try increasing --time or check data.")
            return {day1Name: [], day2Name: []}

        return self._extractRoutes(
            manager, routing, solution,
            day1Name, day1Records, nDay1,
            day2Name, day2Records,
        )

    def _extractRoutes(self, manager, routing, solution,
                       day1Name, day1Records, nDay1, day2Name, day2Records):
        """
        Walk each vehicle's path and split stops back into day-1 and day-2 buckets.
        Node indices 1..nDay1 belong to day-1; nDay1+1..end belong to day-2.
        A vehicle that serves both days has its stops split at the first day-2 node.
        """
        day1Routes = []
        day2Routes = []

        for v in range(routing.vehicles()):
            idx      = routing.Start(v)
            d1Stops  = []
            d2Stops  = []

            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if 1 <= node <= nDay1:
                    d1Stops.append(day1Records[node - 1])
                elif node > nDay1:
                    d2Stops.append(day2Records[node - nDay1 - 1])
                idx = solution.Value(routing.NextVar(idx))

            if d1Stops:
                day1Routes.append(d1Stops)
            if d2Stops:
                day2Routes.append(d2Stops)

        return {day1Name: day1Routes, day2Name: day2Routes}

    def _collectStats(self, routes, elapsed):
        if not routes:
            return {"miles": 0, "routes": 0, "feasible": True, "runtime_s": round(elapsed, 2)}

        totalMiles  = sum(evaluateRoute(r)["total_miles"] for r in routes)
        allFeasible = all(evaluateRoute(r)["overall_feasible"] for r in routes)
        return {
            "miles":     totalMiles,
            "routes":    len(routes),
            "feasible":  allFeasible,
            "runtime_s": round(elapsed, 2),
        }
