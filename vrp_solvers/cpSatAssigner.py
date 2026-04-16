"""
vrp_solvers/cpSatAssigner.py — OR-Tools CP-SAT day assignment for Q3 Option C.

Replaces the greedy angular sweep (Option B) with a formal integer program
solved by CP-SAT.  The model jointly optimises two objectives:

  1. Geographic affinity — maximise the total cube of orders assigned to their
     angularly-preferred day (same sector used by Option B), weighted so that
     high-volume stores contribute more to the objective.

  2. Cube balance — hard bounds keep every day's load within ±25 % of the
     weekly average, preventing CP-SAT from stacking everything on one day.

Additional constraints
  - Every order is assigned to exactly one day.
  - Multi-visit stores: at most ceil(N_store / 5) orders per day per store.
    This handles the 8 Boston-area stores with 6–8 weekly orders gracefully.

The angular bearing pre-computation is reused from AngularSweepAssigner so
Option B and C share the same geographic reference.

Exports: CpSatAssigner
"""

import math
import os

import pandas as pd
from ortools.sat.python import cp_model

from vrp_solvers.base import DATA_DIR, DAYS, DEPOT_ZIP
from vrp_solvers.angularSweepAssigner import (
    AngularSweepAssigner,
    _bearing,
    _DEPOT_LON,
    _DEPOT_LAT,
)

_NUM_DAYS     = len(DAYS)                    # 5
_BALANCE_LO   = 0.75                         # daily cube ≥ 75 % of target
_BALANCE_HI   = 1.25                         # daily cube ≤ 125 % of target
_DEFAULT_TIME = 30                           # CP-SAT wall-clock limit (seconds)


class CpSatAssigner:
    """
    Assigns delivery days via CP-SAT integer programming.
    Interface mirrors AngularSweepAssigner: call assign(), then getSectorSummary().
    """

    def __init__(self, orders, locs, timeLimitSec=_DEFAULT_TIME):
        self._orders      = orders.copy()
        self._locs        = locs.copy()
        self._timeLimitSec = timeLimitSec
        self._summary     = None
        self._solveStatus = None

    def assign(self):
        """
        Return a copy of orders with CP-SAT-optimised DayOfWeek values.
        Falls back to the angular sweep result if CP-SAT cannot find a
        feasible solution within the time limit.
        """
        zipCoords     = self._buildZipCoords()
        preferredDay  = self._computePreferredDays(zipCoords)
        revisedOrders = self._solveAndApply(preferredDay)
        return revisedOrders

    def getSectorSummary(self):
        if self._summary is None:
            raise RuntimeError("Call assign() before getSectorSummary().")
        return self._summary

    def getSolveStatus(self):
        """Return the CP-SAT solver status string from the last solve."""
        return self._solveStatus

    # ------------------------------------------------------------------

    def _buildZipCoords(self):
        coords = {}
        lonCol = "lon" if "lon" in self._locs.columns else "X"
        latCol = "lat" if "lat" in self._locs.columns else "Y"
        for _, row in self._locs.iterrows():
            z = int(row["ZIP"])
            coords[z] = (float(row[lonCol]), float(row[latCol]))
        return coords

    def _computePreferredDays(self, zipCoords):
        """
        Pre-compute a preferred day index (0–4) for every order using the
        same angular-sector logic as AngularSweepAssigner.  This gives
        CP-SAT a geographic warm-start objective to maximise.
        """
        # Reuse AngularSweepAssigner to get the greedy sector assignments
        sweeper = AngularSweepAssigner(self._orders, self._locs)
        swept   = sweeper.assign()

        # Map ORDERID → preferred day index
        dayIndex   = {d: i for i, d in enumerate(DAYS)}
        preferredDay = {}
        for _, row in swept.iterrows():
            oid = int(row["ORDERID"])
            preferredDay[oid] = dayIndex[row["DayOfWeek"]]

        return preferredDay

    def _solveAndApply(self, preferredDay):
        """Build and solve the CP-SAT model; apply the solution to orders."""
        records  = self._orders.to_dict("records")
        n        = len(records)
        cubes    = [int(r["CUBE"]) for r in records]
        orderIds = [int(r["ORDERID"]) for r in records]

        totalCube = sum(cubes)
        target    = totalCube / _NUM_DAYS
        loBound   = int(math.floor(target * _BALANCE_LO))
        hiBound   = int(math.ceil (target * _BALANCE_HI))

        # Group order indices by store ZIP for spread constraints
        storeGroups = {}
        for idx, rec in enumerate(records):
            z = int(rec["TOZIP"])
            storeGroups.setdefault(z, []).append(idx)

        # ── build model ─────────────────────────────────────────────────
        model = cp_model.CpModel()

        # x[i][d] = 1 if order i is assigned to day d
        x = [
            [model.new_bool_var(f"x_{i}_{d}") for d in range(_NUM_DAYS)]
            for i in range(n)
        ]

        # Each order on exactly one day
        for i in range(n):
            model.add_exactly_one(x[i])

        # Daily cube balance — hard bounds
        for d in range(_NUM_DAYS):
            dailyCube = sum(cubes[i] * x[i][d] for i in range(n))
            model.add(dailyCube >= loBound)
            model.add(dailyCube <= hiBound)

        # Multi-visit spread: at most ceil(N/5) orders from same store per day.
        # For most stores N≤5 so maxPerDay=1 (one visit per day).
        # For the 8 high-frequency Boston-area stores (N=6-8) maxPerDay=2.
        for storeZip, indices in storeGroups.items():
            maxPerDay = math.ceil(len(indices) / _NUM_DAYS)
            if maxPerDay >= len(indices):
                # Store has ≥ 5 orders — at-most-1 can't be satisfied, allow 2
                maxPerDay = max(1, maxPerDay)
            for d in range(_NUM_DAYS):
                model.add(sum(x[i][d] for i in indices) <= maxPerDay)

        # ── objective: maximise cube-weighted geographic affinity ────────
        # Orders assigned to their angularly-preferred day contribute their
        # full cube to the objective; misassigned orders contribute zero.
        # This steers CP-SAT toward geographically compact daily clusters.
        affinityTerms = []
        for i, oid in enumerate(orderIds):
            prefD = preferredDay.get(oid, 0)
            # weight = cube so high-volume stores dominate the objective
            affinityTerms.append(cubes[i] * x[i][prefD])

        model.maximize(sum(affinityTerms))

        # ── solve ────────────────────────────────────────────────────────
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self._timeLimitSec
        solver.parameters.num_workers = 4     # use multiple threads for speed
        solver.parameters.log_search_progress = False

        status = solver.solve(model)
        self._solveStatus = solver.status_name(status)

        print(f"  CP-SAT status: {self._solveStatus} "
              f"| objective: {solver.objective_value:.0f} / {sum(cubes):.0f} "
              f"({solver.objective_value / sum(cubes) * 100:.1f}% affinity)")

        # ── extract solution or fall back to sweep ───────────────────────
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._buildRevisedOrders(records, x, solver)
        else:
            print("  CP-SAT found no feasible solution — falling back to angular sweep.")
            sweeper = AngularSweepAssigner(self._orders, self._locs)
            return sweeper.assign()

    def _buildRevisedOrders(self, records, x, solver):
        """Reconstruct the orders DataFrame from the CP-SAT solution."""
        revised  = self._orders.copy()
        newDays  = []
        dayCubes = {d: 0.0 for d in DAYS}

        for i, rec in enumerate(records):
            assignedD = next(
                d for d in range(_NUM_DAYS) if solver.value(x[i][d]) == 1
            )
            newDays.append(DAYS[assignedD])
            dayCubes[DAYS[assignedD]] += rec["CUBE"]

        revised["DayOfWeek"] = newDays

        totalCube = sum(rec["CUBE"] for rec in records)
        target    = totalCube / _NUM_DAYS
        summaryRows = [
            {
                "day":            d,
                "cube_total":     round(dayCubes[d]),
                "cube_target":    round(target),
                "pct_of_target":  round(dayCubes[d] / target * 100, 1),
            }
            for d in DAYS
        ]
        self._summary = pd.DataFrame(summaryRows)

        return revised
