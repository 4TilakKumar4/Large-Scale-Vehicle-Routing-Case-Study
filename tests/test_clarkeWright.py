"""
tests/test_clarkeWright.py — Unit tests for vrp_solvers/clarkeWright.py.
"""

import unittest

from tests.fixtures import injectFixture, MON_DF, TUE_DF, makeOrdersDf, makeOrder

injectFixture()

from vrp_solvers.base import evaluateRoute, VAN_CAPACITY
from vrp_solvers.clarkeWright import ClarkeWrightSolver


class TestClarkeWrightSolverEmptyDay(unittest.TestCase):

    def setUp(self):
        self.solver = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)

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


class TestClarkeWrightSolverValidDay(unittest.TestCase):

    def setUp(self):
        self.solver = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)
        self.routes = self.solver.solve(MON_DF)

    def test_returnsNonEmptyRoutes(self):
        self.assertGreater(len(self.routes), 0)

    def test_allRoutesFeasible(self):
        for route in self.routes:
            self.assertTrue(evaluateRoute(route)["overall_feasible"])

    def test_allOrdersCovered(self):
        # Every order ID in MON_DF must appear exactly once across all routes
        routed = [int(stop["ORDERID"]) for route in self.routes for stop in route]
        expected = sorted(MON_DF["ORDERID"].astype(int).tolist())
        self.assertEqual(sorted(routed), expected)

    def test_noOrderDuplicated(self):
        routed = [int(stop["ORDERID"]) for route in self.routes for stop in route]
        self.assertEqual(len(routed), len(set(routed)))

    def test_statsRoutesMatchActual(self):
        stats = self.solver.getStats()
        self.assertEqual(stats["routes"], len(self.routes))

    def test_statsMilesPositive(self):
        stats = self.solver.getStats()
        self.assertGreater(stats["miles"], 0)

    def test_statsFeasibleTrue(self):
        stats = self.solver.getStats()
        self.assertTrue(stats["feasible"])

    def test_convergenceNone(self):
        self.assertIsNone(self.solver.getConvergence())


class TestClarkeWrightSolverNoLocalSearch(unittest.TestCase):

    def test_constructionOnlyAllOrdersCovered(self):
        solver  = ClarkeWrightSolver(useTwoOpt=False, useOrOpt=False)
        routes  = solver.solve(MON_DF)
        routed  = [int(stop["ORDERID"]) for route in routes for stop in route]
        expected = sorted(MON_DF["ORDERID"].astype(int).tolist())
        self.assertEqual(sorted(routed), expected)

    def test_localSearchReducesMilesOrEqual(self):
        noLS    = ClarkeWrightSolver(useTwoOpt=False, useOrOpt=False)
        withLS  = ClarkeWrightSolver(useTwoOpt=True,  useOrOpt=True)
        milesNoLS   = noLS.solve(MON_DF);   statsNoLS   = noLS.getStats()
        milesWithLS = withLS.solve(MON_DF); statsWithLS = withLS.getStats()
        self.assertLessEqual(statsWithLS["miles"], statsNoLS["miles"])


if __name__ == "__main__":
    unittest.main()
