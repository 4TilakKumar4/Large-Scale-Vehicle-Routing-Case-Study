"""
vrp_solvers/angularSweepAssigner.py — Angular-sector day assignment for Q3 Option B.

Reassigns store orders to delivery days by partitioning stores into five
angular sectors radiating from the Wilmington depot.  Sector boundaries are
placed so that each day's total cube load is as close to the weekly average
as possible (cube-balanced sweep).

Algorithm
---------
1. Compute the bearing of every store from the depot using its (lon, lat).
2. Sort stores by bearing, creating a naturally geographic ordering.
3. Walk stores in bearing order, accumulating cube into the current day bucket.
   When a bucket reaches the per-day target, advance to the next day.
4. For multi-visit stores (>1 order in a week):
     - First order  → the angular sector's assigned day.
     - Extra orders → distributed round-robin across the four adjacent days,
       choosing the day with the lowest cube load at each step.
5. Return a copy of the orders DataFrame with updated DayOfWeek values.

The reassigned orders are then fed directly into ORToolsOvernightSolver,
so the routing layer is identical to Option A.

Exports: AngularSweepAssigner
"""

import math
import os

import pandas as pd

from vrp_solvers.base import DATA_DIR, DAYS, DEPOT_ZIP

# Depot coordinates — Wilmington MA (ZIP 01887)
_DEPOT_LON = -71.17555556
_DEPOT_LAT =  42.54555556


def _bearing(lon, lat):
    """
    Compass bearing in [0, 360) from depot to (lon, lat).
    Uses flat-earth approximation — accurate enough for the ~200-mile NE region.
    """
    dx = lon - _DEPOT_LON
    dy = lat - _DEPOT_LAT
    # atan2 measured from east; convert to compass bearing from north
    angle = math.degrees(math.atan2(dx, dy))
    return angle % 360


class AngularSweepAssigner:
    """
    Reassigns delivery days via angular-sector partitioning.
    Call assign() to get a revised orders DataFrame.
    Call getSectorSummary() after assign() for diagnostics.
    """

    def __init__(self, orders, locs):
        """
        orders : cleaned orders DataFrame (ORDERID, TOZIP, CUBE, DayOfWeek, …)
        locs   : locations DataFrame with columns ZIP, lon, lat
        """
        self._orders = orders.copy()
        self._locs   = locs.copy()
        self._summary = None

    def assign(self):
        """
        Return a copy of orders with DayOfWeek reassigned by angular sector.
        Original order IDs, cubes, and ZIP codes are preserved — only
        DayOfWeek changes.
        """
        zipCoords  = self._buildZipCoords()
        storeBearing = self._computeStoreBearings(zipCoords)
        dayMap     = self._sweepAssign(storeBearing)
        return self._applyDayMap(dayMap)

    def getSectorSummary(self):
        """
        Return a DataFrame showing cube totals and store counts per assigned day.
        Only valid after assign() has been called.
        """
        if self._summary is None:
            raise RuntimeError("Call assign() before getSectorSummary().")
        return self._summary

    # ------------------------------------------------------------------

    def _buildZipCoords(self):
        """Build {zip: (lon, lat)} from the locations table."""
        coords = {}
        for _, row in self._locs.iterrows():
            z = int(row["ZIP"])
            # locations_clean.csv uses lon/lat column names after VRP_DataAnalysis
            lonCol = "lon" if "lon" in self._locs.columns else "X"
            latCol = "lat" if "lat" in self._locs.columns else "Y"
            coords[z] = (float(row[lonCol]), float(row[latCol]))
        return coords

    def _computeStoreBearings(self, zipCoords):
        """
        Return a list of (bearing, storeZip, totalCube) sorted by bearing.
        Stores without coordinates are placed at bearing=0 (rare edge case).
        """
        storeStats = (
            self._orders
            .groupby("TOZIP")["CUBE"]
            .agg(totalCube="sum", numOrders="count")
            .reset_index()
        )

        rows = []
        for _, row in storeStats.iterrows():
            z         = int(row["TOZIP"])
            totalCube = float(row["totalCube"])
            numOrders = int(row["numOrders"])
            if z in zipCoords and z != DEPOT_ZIP:
                lon, lat = zipCoords[z]
                b = _bearing(lon, lat)
            else:
                b = 0.0
            rows.append((b, z, totalCube, numOrders))

        rows.sort(key=lambda x: x[0])
        return rows

    def _sweepAssign(self, storeBearing):
        """
        Core sweep: walk stores in angular order, fill day buckets by cube target.
        Returns {storeZip: [day, day, …]} where the list has one entry per order.
        """
        totalCube   = sum(r[2] for r in storeBearing)
        targetPerDay = totalCube / len(DAYS)

        # Running cube totals per day
        dayCube  = {d: 0.0 for d in DAYS}
        dayMap   = {}          # storeZip → list of assigned days (one per order)

        dayIdx    = 0          # current day bucket index

        for bearing, storeZip, storeCube, numOrders in storeBearing:
            primaryDay = DAYS[dayIdx]

            # Advance to next day bucket when this one is full —
            # but never overflow past the last day
            if (dayCube[primaryDay] + storeCube > targetPerDay * 1.15
                    and dayIdx < len(DAYS) - 1):
                dayIdx    += 1
                primaryDay = DAYS[dayIdx]

            # First order always goes to the primary (angular) day
            assignedDays = [primaryDay]
            dayCube[primaryDay] += storeCube / numOrders

            # Extra orders for multi-visit stores: assign greedily to the
            # day with the lowest current cube load, excluding the primary day
            # so we don't pile everything on one day.
            for _ in range(numOrders - 1):
                otherDays  = [d for d in DAYS if d != primaryDay]
                cheapestDay = min(otherDays, key=lambda d: dayCube[d])
                assignedDays.append(cheapestDay)
                dayCube[cheapestDay] += storeCube / numOrders

            dayMap[storeZip] = assignedDays

        # Build summary for diagnostics
        summaryRows = []
        for d in DAYS:
            summaryRows.append({
                "day":        d,
                "cube_total": round(dayCube[d]),
                "cube_target": round(targetPerDay),
                "pct_of_target": round(dayCube[d] / targetPerDay * 100, 1),
            })
        self._summary = pd.DataFrame(summaryRows)

        return dayMap

    def _applyDayMap(self, dayMap):
        """
        Apply the day map back onto the orders DataFrame.
        For each store, pop from its assigned-days list in order of ORDERID
        so the assignment is deterministic.
        """
        revised = self._orders.copy()

        # Track consumption index per store
        consumeIdx = {z: 0 for z in dayMap}

        newDays = []
        for _, row in revised.iterrows():
            z   = int(row["TOZIP"])
            if z in dayMap and consumeIdx[z] < len(dayMap[z]):
                newDay = dayMap[z][consumeIdx[z]]
                consumeIdx[z] += 1
            else:
                # Fallback: keep original if store not in map (depot row guard)
                newDay = row["DayOfWeek"]
            newDays.append(newDay)

        revised["DayOfWeek"] = newDays
        return revised


def loadAngularSweepInputs():
    """Load orders and locations from data/; return (orders, locs)."""
    ordersPath = os.path.join(DATA_DIR, "orders_clean.csv")
    locsPath   = os.path.join(DATA_DIR, "locations_clean.csv")

    orders = pd.read_csv(ordersPath)
    locs   = pd.read_csv(locsPath)

    orders["CUBE"]    = pd.to_numeric(orders["CUBE"],    errors="raise")
    orders["TOZIP"]   = pd.to_numeric(orders["TOZIP"],   errors="raise")
    orders["ORDERID"] = pd.to_numeric(orders["ORDERID"], errors="raise")
    locs["ZIP"]       = pd.to_numeric(locs["ZIP"],       errors="coerce")

    return orders, locs
