"""
tests/test_tabuSearch.py — Unit tests for vrp_solvers/tabuSearch.py.
"""

import unittest

from tests.fixtures import injectFixture, MON_DF, makeOrdersDf

injectFixture()

from vrp_solvers.base import evaluateRoute
from vrp_solvers.clarkeWright import ClarkeWrightSolver
from vrp_solvers.tabuSearch import TabuSearchSolver


class TestTabuSearchEmptyDay(unittest.TestCase):

    def setUp(self):
        self.solver = TabuSearchSolver(maxIter=10)

    def test_emptyDayReturnsEmptyList(self):
        routes = self.solver.solve(makeOrdersDf([]))
        self.assertEqual(routes, [])

    def test_emptyDayStatsZeroed(self):
        self.solver.solve(makeOrdersDf([]))
        stats = self.solver.getStats()
        self.assertEqual(stats["miles"],  0)
        self.assertEqual(stats["routes"], 0)
        self.assertTrue(stats["feasible"])

    def test_emptyDayConvergenceEmpty(self):
        self.solver.solve(makeOrdersDf([]))
        self.assertEqual(self.solver.getConvergence(), [])


class TestTabuSearchValidDay(unittest.TestCase):

    def setUp(self):
        # Small iteration count so tests run quickly
        self.solver = TabuSearchSolver(maxIter=20, tabuTenure=5, randomSeed=42)
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

    def test_milesNotWorseThanSeed(self):
        # TS seeds from CW + local search; result should be ≤ seed miles
        cwMiles = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True).solve(MON_DF)
        cwStats = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)
        cwStats.solve(MON_DF)
        seedMiles = cwStats.getStats()["miles"]
        tsMiles   = self.solver.getStats()["miles"]
        self.assertLessEqual(tsMiles, seedMiles)

    def test_convergenceHasCorrectLength(self):
        # Convergence list has one entry per iteration plus the initial value
        conv = self.solver.getConvergence()
        self.assertEqual(len(conv), 21)   # maxIter=20 + initial

    def test_convergenceMonotonicallyNonIncreasing(self):
        conv = self.solver.getConvergence()
        for i in range(1, len(conv)):
            self.assertLessEqual(conv[i], conv[i - 1])

    def test_statsRuntimeNonNegative(self):
        self.assertGreaterEqual(self.solver.getStats()["runtime_s"], 0)


if __name__ == "__main__":
    unittest.main()