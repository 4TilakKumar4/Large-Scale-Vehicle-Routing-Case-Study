"""
vrp_solvers/orToolsSolver.py — Google OR-Tools CVRP (Q1: homogeneous vans, time windows + HOS).

Implements a RoutingModel with four dimensions:
  Capacity  — cubic-foot load cap per van
  Time      — clock-time with delivery windows [08:00, 18:00]
  DriveTime — cumulative driving minutes, capped at MAX_DRIVING (660 min)
  DutyTime  — cumulative on-duty minutes (drive + service), capped at MAX_DUTY (840 min)

Designed as a drop-in comparison solver alongside ClarkeWrightSolver and ALNS.
Call solve() per day; retrieve results with getStats() and getConvergence().
"""

import math
import time

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from vrp_solvers.base import (
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

# All OR-Tools dimensions must be integers. Time unit: minutes.
_MINS_PER_HOUR = 60
_OPEN_MIN      = int(WINDOW_OPEN  * _MINS_PER_HOUR)   # 480
_CLOSE_MIN     = int(WINDOW_CLOSE * _MINS_PER_HOUR)   # 1080
_MAX_DRIVE_MIN = int(MAX_DRIVING  * _MINS_PER_HOUR)   # 660
_MAX_DUTY_MIN  = int(MAX_DUTY     * _MINS_PER_HOUR)   # 840

# Generous upper bound so the time dimension never falsely cuts feasible routes.
# Set to CLOSE + MAX_DUTY rather than a fixed horizon so late-returning trucks
# aren't penalised by an arbitrarily tight ceiling.
_HORIZON = _CLOSE_MIN + _MAX_DUTY_MIN                 # 1920


def _toMins(miles):
    """Road miles → integer travel minutes at DRIVING_SPEED mph."""
    return int(round(miles * _MINS_PER_HOUR / DRIVING_SPEED))


def _serviceMins(cube):
    """Unload time in integer minutes, respecting the MIN_TIME floor."""
    return int(round(max(MIN_TIME, UNLOAD_RATE * cube)))


class ORToolsSolver:
    """
    Google OR-Tools CVRP with capacity, time-window, and DOT HOS constraints.
    Four dimensions: Capacity, Time, DriveTime, DutyTime.
    Uses PARALLEL_CHEAPEST_INSERTION + GUIDED_LOCAL_SEARCH.
    """

    def __init__(self, timeLimitSec=30, extraVehicles=2):
        self.timeLimitSec = timeLimitSec
        self.extraVehicles = extraVehicles
        self._stats = None

    def solve(self, dayOrders):
        """Build routes for one day and return a list of order-record lists."""
        if dayOrders.empty:
            print("  ORToolsSolver: no orders for this day, skipping.")
            self._stats = {"miles": 0, "routes": 0, "feasible": True, "runtime_s": 0.0}
            return []

        t0      = time.time()
        records = dayOrders.to_dict("records")
        routes  = self._buildRoutes(records)

        self._stats = self._collectStats(routes, time.time() - t0)
        return routes

    def getStats(self):
        """Return miles, routes, runtime, and feasibility from the last solve() call."""
        return self._stats

    def getConvergence(self):
        """OR-Tools does not expose per-iteration convergence data."""
        return None

    # ------------------------------------------------------------------

    def _buildRoutes(self, records):
        n        = len(records)
        nodeZip  = [DEPOT_ZIP] + [r["TOZIP"]  for r in records]
        nodeCube = [0]         + [r["CUBE"]   for r in records]

        # Pre-compute the travel-time matrix once to avoid repeated dict lookups
        # inside the Python callbacks — callbacks fire O(n²) times per iteration.
        travelMin = [
            [
                0 if i == j else _toMins(getDistance(nodeZip[i], nodeZip[j]))
                for j in range(n + 1)
            ]
            for i in range(n + 1)
        ]
        serviceMin = [0] + [_serviceMins(r["CUBE"]) for r in records]

        # Three independent lower bounds on vehicles — take the max.
        # Capacity bound: total cube / van capacity.
        # HOS duty bound: total service time alone may exceed one vehicle's 840-min budget.
        # Stop-density bound: HOS limits each route to ~8 stops on a busy day,
        # so n/8 guards against running out of vehicles on large days like Tuesday.
        totalCube       = sum(r["CUBE"] for r in records)
        totalServiceMin = sum(_serviceMins(r["CUBE"]) for r in records)

        capBound  = math.ceil(totalCube       / VAN_CAPACITY)
        hosBound  = math.ceil(totalServiceMin / _MAX_DUTY_MIN)
        stopBound = math.ceil(len(records)    / 8)

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
            0,                           # no slack — capacity is hard
            [VAN_CAPACITY] * numVehicles,
            True,                        # start cumul at zero
            "Capacity",
        )

        # ── time dimension: transit = travel + service at from-node ────
        # Service time is attributed to the departing node so the cumul at
        # node i reflects the time the vehicle is *free to leave*, which
        # maps cleanly onto the delivery-window check in evaluateRoute().
        def timeCb(fromIdx, toIdx):
            fromNode = manager.IndexToNode(fromIdx)
            toNode   = manager.IndexToNode(toIdx)
            return travelMin[fromNode][toNode] + serviceMin[fromNode]

        timeCbIdx = routing.RegisterTransitCallback(timeCb)
        routing.AddDimension(
            timeCbIdx,
            _OPEN_MIN,   # max waiting at a node before the window opens
            _HORIZON,    # max cumulative time per vehicle
            False,       # departure time is a free variable per vehicle
            "Time",
        )
        timeDim = routing.GetDimensionOrDie("Time")

        # Pin every store node to the delivery window; depot is unconstrained.
        for i in range(1, n + 1):
            timeDim.CumulVar(manager.NodeToIndex(i)).SetRange(_OPEN_MIN, _CLOSE_MIN)

        # ── drive-time dimension (travel only) ──────────────────────────
        # Reuses distCbIdx so only pure driving accumulates — service time
        # does not count toward the 11-hour drive cap, matching evaluateRoute().
        routing.AddDimension(
            distCbIdx,
            0,               # no slack — drive time is a hard DOT limit
            _MAX_DRIVE_MIN,  # 660 min hard cap
            True,            # reset to zero at depot start
            "DriveTime",
        )

        # ── duty-time dimension (drive + service) ───────────────────────
        # Reuses timeCbIdx (travel + service at from-node) to mirror the
        # totalDuty = totalDrive + totalUnload calculation in evaluateRoute().
        # Wait time is excluded — conservative but consistent with base.py.
        routing.AddDimension(
            timeCbIdx,
            0,              # no slack
            _MAX_DUTY_MIN,  # 840 min hard cap
            True,           # reset to zero at depot start
            "DutyTime",
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
            print("  ORToolsSolver: solver returned no solution — check data or increase time limit.")
            return []

        return self._extractRoutes(manager, routing, solution, records)

    def _extractRoutes(self, manager, routing, solution, records):
        """Walk each vehicle's path and reconstruct order-record lists."""
        routes = []
        for v in range(routing.vehicles()):
            idx   = routing.Start(v)
            route = []
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node != 0:
                    route.append(records[node - 1])
                idx = solution.Value(routing.NextVar(idx))
            if route:
                routes.append(route)
        return routes

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