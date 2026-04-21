"""
tests/test_nearestNeighbor.py — Unit tests for vrp_solvers/nearestNeighbor.py.
"""

import unittest

from tests.fixtures import injectFixture, MON_DF, makeOrdersDf, makeOrder, DEPOT

injectFixture()

from vrp_solvers.base import evaluateRoute, VAN_CAPACITY
from vrp_solvers.nearestNeighbor import NearestNeighborSolver


class TestNearestNeighborEmptyDay(unittest.TestCase):

    def setUp(self):
        self.solver = NearestNeighborSolver(useTwoOpt=True, useOrOpt=True)

    def test_emptyDayReturnsEmptyList(self):
        routes = self.solver.solve(makeOrdersDf([]))
        self.assertEqual(routes, [])

    def test_emptyDayStatsZeroed(self):
        self.solver.solve(makeOrdersDf([]))
        stats = self.solver.getStats()
        self.assertEqual(stats["miles"],  0)
        self.assertEqual(stats["routes"], 0)
        self.assertTrue(stats["feasible"])

    def test_emptyDayConvergenceNone(self):
        self.solver.solve(makeOrdersDf([]))
        self.assertIsNone(self.solver.getConvergence())


class TestNearestNeighborValidDay(unittest.TestCase):

    def setUp(self):
        self.solver = NearestNeighborSolver(useTwoOpt=True, useOrOpt=True)
        self.routes = self.solver.solve(MON_DF)

    def test_returnsNonEmptyRoutes(self):
        self.assertGreater(len(self.routes), 0)

    def test_allRoutesFeasible(self):
        for route in self.routes:
            self.assertTrue(evaluateRoute(route)["overall_feasible"])

    def test_allOrdersCovered(self):
        routed   = [int(stop["ORDERID"]) for route in self.routes for stop in route]
        expected = sorted(MON_DF["ORDERID"].astype(int).tolist())
        self.assertEqual(sorted(routed), expected)

    def test_noOrderDuplicated(self):
        routed = [int(stop["ORDERID"]) for route in self.routes for stop in route]
        self.assertEqual(len(routed), len(set(routed)))

    def test_statsRoutesMatchActual(self):
        stats = self.solver.getStats()
        self.assertEqual(stats["routes"], len(self.routes))

    def test_statsMilesPositive(self):
        self.assertGreater(self.solver.getStats()["miles"], 0)


class TestNearestNeighborOverCapacityGuard(unittest.TestCase):

    def test_overCapacityOrderSkipped(self):
        # Order with cube > VAN_CAPACITY must be skipped, not cause infinite loop
        bigOrder = makeOrder(99, 1001, VAN_CAPACITY + 1, "Mon")
        df       = makeOrdersDf([bigOrder])
        solver   = NearestNeighborSolver()
        routes   = solver.solve(df)
        # The oversized order should not appear in any route
        routed = [int(stop["ORDERID"]) for route in routes for stop in route]
        self.assertNotIn(99, routed)

    def test_normalOrdersStillRoutedWhenOneOverCapacity(self):
        # Mix one oversized order with valid ones — valid ones must still be routed
        orders = [
            makeOrder(1,  1001, 500,             "Mon"),
            makeOrder(99, 1002, VAN_CAPACITY + 1, "Mon"),
            makeOrder(2,  1003, 600,             "Mon"),
        ]
        df     = makeOrdersDf(orders)
        solver = NearestNeighborSolver()
        routes = solver.solve(df)
        routed = [int(stop["ORDERID"]) for route in routes for stop in route]
        self.assertIn(1,  routed)
        self.assertIn(2,  routed)
        self.assertNotIn(99, routed)


class TestNearestNeighborInfeasibleStopGuard(unittest.TestCase):

    def test_individuallyInfeasibleStopDoesNotHang(self):
        # Construct an order whose single-stop route would violate the window —
        # drive time so large the van can't reach and return within [8, 18].
        # We simulate this by creating a fake distance that is very far.
        import vrp_solvers.base as base
        import pandas as pd

        originalMatrix = base.DIST_MATRIX.copy()

        # Add a ZIP 9001 that is 300 miles away from depot — drive alone = 7.5h,
        # so arrival = 8:00 + 7.5 = 15:30, departure = 16:00+, return = 23:30+ → infeasible
        farZip = 9001
        newRow = pd.Series({z: 300.0 for z in base.DIST_MATRIX.columns}, name=farZip)
        newCol = pd.Series({z: 300.0 for z in base.DIST_MATRIX.index},   name=farZip)
        newCol[farZip] = 0.0
        newRow[farZip] = 0.0

        base.DIST_MATRIX = pd.concat([base.DIST_MATRIX, newRow.to_frame().T])
        base.DIST_MATRIX[farZip] = newCol

        try:
            farOrder = makeOrder(77, farZip, 100, "Mon")
            df       = makeOrdersDf([farOrder])
            solver   = NearestNeighborSolver()
            # Should complete without hanging
            routes   = solver.solve(df)
            routed   = [int(stop["ORDERID"]) for route in routes for stop in route]
            self.assertNotIn(77, routed)
        finally:
            base.DIST_MATRIX = originalMatrix


if __name__ == "__main__":
    unittest.main()
