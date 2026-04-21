"""
tests/test_base.py — Unit tests for vrp_solvers/base.py.

Covers: evaluateRoute, getDistance, toClock, consolidateRoutes, applyLocalSearch.
"""

import unittest

from tests.fixtures import injectFixture, MON_ORDERS, makeOrder, DEPOT

injectFixture()

import vrp_solvers.base as base
from vrp_solvers.base import (
    evaluateRoute,
    getDistance,
    toClock,
    consolidateRoutes,
    applyLocalSearch,
    routeIds,
)


class TestToClock(unittest.TestCase):

    def test_wholeHour(self):
        self.assertEqual(toClock(8.0), "08:00")

    def test_halfHour(self):
        self.assertEqual(toClock(8.5), "08:30")

    def test_minuteRollover(self):
        # 8 hours 59.5 minutes rounds to 09:00
        self.assertEqual(toClock(8 + 59.5 / 60), "09:00")

    def test_midnight(self):
        self.assertEqual(toClock(0.0), "00:00")


class TestGetDistance(unittest.TestCase):

    def test_knownDistance(self):
        self.assertEqual(getDistance(DEPOT, 1001), 10.0)

    def test_symmetric(self):
        self.assertEqual(getDistance(1001, 1002), getDistance(1002, 1001))

    def test_selfDistance(self):
        self.assertEqual(getDistance(DEPOT, DEPOT), 0.0)

    def test_unknownZipRaisesKeyError(self):
        with self.assertRaises(KeyError):
            getDistance(DEPOT, 9999)

    def test_beforeLoadInputsRaisesRuntimeError(self):
        # Temporarily unset the matrix and verify the guard fires
        original = base.DIST_MATRIX
        base.DIST_MATRIX = None
        try:
            with self.assertRaises(RuntimeError):
                getDistance(DEPOT, 1001)
        finally:
            base.DIST_MATRIX = original


class TestEvaluateRoute(unittest.TestCase):

    def setUp(self):
        self.singleStop = [MON_ORDERS[0]]        # ZIP 1001, cube 500
        self.twoStop    = MON_ORDERS[:2]          # ZIP 1001 + 1002, cube 1100

    def test_singleStopFeasible(self):
        result = evaluateRoute(self.singleStop)
        self.assertTrue(result["overall_feasible"])

    def test_singleStopMilesPositive(self):
        result = evaluateRoute(self.singleStop)
        self.assertGreater(result["total_miles"], 0)

    def test_singleStopMilesCorrect(self):
        # depot→1001→depot = 10 + 10 = 20 miles
        result = evaluateRoute(self.singleStop)
        self.assertEqual(result["total_miles"], 20)

    def test_twoStopFeasible(self):
        result = evaluateRoute(self.twoStop)
        self.assertTrue(result["overall_feasible"])

    def test_cubeAccumulates(self):
        result = evaluateRoute(self.twoStop)
        self.assertEqual(result["total_cube"], 1100)

    def test_emptyRouteRaisesValueError(self):
        with self.assertRaises(ValueError):
            evaluateRoute([])

    def test_overCapacityInfeasible(self):
        # Single stop whose cube exceeds VAN_CAPACITY
        bigOrder = [makeOrder(99, 1001, base.VAN_CAPACITY + 1, "Mon")]
        result   = evaluateRoute(bigOrder)
        self.assertFalse(result["capacity_feasible"])
        self.assertFalse(result["overall_feasible"])

    def test_returnTimeAfterDispatch(self):
        result = evaluateRoute(self.singleStop)
        self.assertGreater(result["return_time"], 0)

    def test_driveTimeLessThanMaxDriving(self):
        result = evaluateRoute(self.singleStop)
        self.assertLessEqual(result["total_drive"], base.MAX_DRIVING)


class TestConsolidateRoutes(unittest.TestCase):

    def test_emptyInputReturnsEmpty(self):
        self.assertEqual(consolidateRoutes([]), [])

    def test_singleRouteUnchanged(self):
        route  = [MON_ORDERS[0]]
        result = consolidateRoutes([route])
        self.assertEqual(len(result), 1)

    def test_consolidatesWhenPossible(self):
        # Two single-stop routes whose combined cube fits in one van
        r1 = [MON_ORDERS[0]]  # cube 500
        r2 = [MON_ORDERS[1]]  # cube 600 — combined 1100, fits in 3200
        result = consolidateRoutes([r1, r2])
        # Should merge into one route
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)


class TestApplyLocalSearch(unittest.TestCase):

    def test_emptyInputReturnsEmpty(self):
        self.assertEqual(applyLocalSearch([]), [])

    def test_singleStopRouteUnchanged(self):
        route  = [MON_ORDERS[0]]
        result = applyLocalSearch([route])
        self.assertEqual(len(result), 1)
        self.assertEqual(routeIds(result[0]), [1])

    def test_multiStopRouteMilesNotWorse(self):
        route        = list(MON_ORDERS)
        beforeMiles  = evaluateRoute(route)["total_miles"]
        improved     = applyLocalSearch([route])
        afterMiles   = evaluateRoute(improved[0])["total_miles"]
        self.assertLessEqual(afterMiles, beforeMiles)


if __name__ == "__main__":
    unittest.main()
