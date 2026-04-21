"""
tests/test_alns.py — Unit tests for vrp_solvers/alns.py.
"""

import unittest

from tests.fixtures import injectFixture, MON_DF, makeOrdersDf

injectFixture()

from vrp_solvers.base import evaluateRoute
from vrp_solvers.clarkeWright import ClarkeWrightSolver
from vrp_solvers.alns import ALNSSolver


class TestALNSEmptyDay(unittest.TestCase):

    def setUp(self):
        self.solver = ALNSSolver(maxIter=10)

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

    def test_emptyDayWeightHistoryEmpty(self):
        self.solver.solve(makeOrdersDf([]))
        self.assertEqual(self.solver.getWeightHistory(), {})


class TestALNSValidDay(unittest.TestCase):

    def setUp(self):
        self.solver = ALNSSolver(maxIter=30, removeFrac=0.3, randomSeed=42)
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
        alnsMiles = self.solver.getStats()["miles"]
        self.assertLessEqual(alnsMiles, seedMiles)

    def test_convergenceLengthCorrect(self):
        conv = self.solver.getConvergence()
        # maxIter=30 + 1 initial entry
        self.assertEqual(len(conv), 31)

    def test_convergenceMonotonicallyNonIncreasing(self):
        conv = self.solver.getConvergence()
        for i in range(1, len(conv)):
            self.assertLessEqual(conv[i], conv[i - 1])

    def test_weightHistoryHasCorrectKeys(self):
        wh = self.solver.getWeightHistory()
        self.assertIn("destroy", wh)
        self.assertIn("repair",  wh)

    def test_weightHistoryDestroyOperatorNames(self):
        destroyNames = list(self.solver.getWeightHistory()["destroy"].keys())
        self.assertIn("random", destroyNames)
        self.assertIn("worst",  destroyNames)
        self.assertIn("shaw",   destroyNames)
        self.assertIn("route",  destroyNames)

    def test_weightHistoryRepairOperatorNames(self):
        repairNames = list(self.solver.getWeightHistory()["repair"].keys())
        self.assertIn("greedy", repairNames)
        self.assertIn("regret", repairNames)

    def test_weightHistoryLengthMatchesMaxIter(self):
        wh = self.solver.getWeightHistory()
        for name, curve in wh["destroy"].items():
            self.assertEqual(len(curve), 30, msg=f"destroy/{name} wrong length")
        for name, curve in wh["repair"].items():
            self.assertEqual(len(curve), 30, msg=f"repair/{name} wrong length")

    def test_weightsAllPositive(self):
        wh = self.solver.getWeightHistory()
        for name, curve in wh["destroy"].items():
            for v in curve:
                self.assertGreater(v, 0, msg=f"destroy/{name} has non-positive weight")
        for name, curve in wh["repair"].items():
            for v in curve:
                self.assertGreater(v, 0, msg=f"repair/{name} has non-positive weight")

    def test_statsRuntimeNonNegative(self):
        self.assertGreaterEqual(self.solver.getStats()["runtime_s"], 0)


if __name__ == "__main__":
    unittest.main()