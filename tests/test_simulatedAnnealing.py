"""
tests/test_simulatedAnnealing.py — Unit tests for vrp_solvers/simulatedAnnealing.py.
"""

import unittest

from tests.fixtures import injectFixture, MON_DF, makeOrdersDf

injectFixture()

from vrp_solvers.base import evaluateRoute
from vrp_solvers.clarkeWright import ClarkeWrightSolver
from vrp_solvers.simulatedAnnealing import SimulatedAnnealingSolver


class TestSAEmptyDay(unittest.TestCase):

    def setUp(self):
        self.solver = SimulatedAnnealingSolver(maxIter=10)

    def test_emptyDayReturnsEmptyList(self):
        self.assertEqual(self.solver.solve(makeOrdersDf([])), [])

    def test_emptyDayStatsZeroed(self):
        self.solver.solve(makeOrdersDf([]))
        stats = self.solver.getStats()
        self.assertEqual(stats["miles"],  0)
        self.assertEqual(stats["routes"], 0)
        self.assertTrue(stats["feasible"])

    def test_emptyDayConvergenceEmpty(self):
        self.solver.solve(makeOrdersDf([]))
        self.assertEqual(self.solver.getConvergence(), [])


class TestSAValidDay(unittest.TestCase):

    def setUp(self):
        self.solver = SimulatedAnnealingSolver(
            maxIter=50, tempStart=200.0, tempEnd=1.0, randomSeed=42
        )
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
        cw = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)
        cw.solve(MON_DF)
        seedMiles = cw.getStats()["miles"]
        saMiles   = self.solver.getStats()["miles"]
        self.assertLessEqual(saMiles, seedMiles)

    def test_convergenceLengthCorrect(self):
        conv = self.solver.getConvergence()
        # maxIter=50 iterations + 1 initial value
        self.assertEqual(len(conv), 51)

    def test_convergenceMonotonicallyNonIncreasing(self):
        conv = self.solver.getConvergence()
        for i in range(1, len(conv)):
            self.assertLessEqual(conv[i], conv[i - 1])

    def test_convergenceAllPositive(self):
        for v in self.solver.getConvergence():
            self.assertGreater(v, 0)

    def test_statsRuntimeNonNegative(self):
        self.assertGreaterEqual(self.solver.getStats()["runtime_s"], 0)


if __name__ == "__main__":
    unittest.main()