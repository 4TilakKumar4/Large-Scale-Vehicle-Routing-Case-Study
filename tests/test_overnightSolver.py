"""
tests/test_overnightSolver.py — Unit tests for vrp_solvers/overnightSolver.py.
"""

import unittest

from tests.fixtures import (
    injectFixture,
    MON_ORDERS, TUE_ORDERS,
    MON_DF, TUE_DF, ALL_DF,
    makeOrdersDf,
)

injectFixture()

from vrp_solvers.base import evaluateRoute, DAYS
from vrp_solvers.clarkeWright import ClarkeWrightSolver
from vrp_solvers.overnightSolver import (
    OvernightSolver,
    evaluateOvernightRoute,
    applyOvernightImprovements,
    serveRouteSegment,
)


class TestEvaluateOvernightRoute(unittest.TestCase):

    def test_emptyDay1ReturnInfeasible(self):
        result = evaluateOvernightRoute([], TUE_ORDERS)
        self.assertFalse(result["overall_feasible"])

    def test_emptyDay2ReturnInfeasible(self):
        result = evaluateOvernightRoute(MON_ORDERS, [])
        self.assertFalse(result["overall_feasible"])

    def test_validPairReturnsFeasibility(self):
        # Use single-stop routes so they definitely fit within HOS
        r1 = [MON_ORDERS[0]]  # one stop Mon
        r2 = [TUE_ORDERS[0]]  # one stop Tue
        result = evaluateOvernightRoute(r1, r2)
        # Result should have the key even if infeasible
        self.assertIn("overall_feasible", result)

    def test_resultContainsRequiredKeys(self):
        r1 = [MON_ORDERS[0]]
        r2 = [TUE_ORDERS[0]]
        result = evaluateOvernightRoute(r1, r2)
        for key in ["total_miles", "day1_drive", "day2_drive",
                    "break_start_time", "break_end_time",
                    "window_feasible", "overall_feasible"]:
            self.assertIn(key, result)

    def test_totalMilesPositiveWhenFeasible(self):
        r1 = [MON_ORDERS[0]]
        r2 = [TUE_ORDERS[0]]
        result = evaluateOvernightRoute(r1, r2)
        if result["overall_feasible"]:
            self.assertGreater(result["total_miles"], 0)

    def test_breakEndAfterBreakStart(self):
        r1 = [MON_ORDERS[0]]
        r2 = [TUE_ORDERS[0]]
        result = evaluateOvernightRoute(r1, r2)
        if result["overall_feasible"]:
            self.assertGreater(result["break_end_time"], result["break_start_time"])


class TestApplyOvernightImprovements(unittest.TestCase):

    def _buildRoutesByDay(self):
        solver = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)
        routesByDay = {}
        for day in ["Mon", "Tue"]:
            df = ALL_DF[ALL_DF["DayOfWeek"] == day].copy()
            routesByDay[day] = solver.solve(df)
        # Fill remaining days with empty lists so the dict is complete
        for day in DAYS:
            routesByDay.setdefault(day, [])
        return routesByDay

    def test_returnsCorrectTypes(self):
        routesByDay = self._buildRoutesByDay()
        overnightRoutes, usedRoutes = applyOvernightImprovements(routesByDay)
        self.assertIsInstance(overnightRoutes, list)
        self.assertIsInstance(usedRoutes, dict)

    def test_usedRoutesHasAllDays(self):
        routesByDay = self._buildRoutesByDay()
        _, usedRoutes = applyOvernightImprovements(routesByDay)
        for day in routesByDay:
            self.assertIn(day, usedRoutes)

    def test_noRouteUsedTwice(self):
        routesByDay = self._buildRoutesByDay()
        overnightRoutes, usedRoutes = applyOvernightImprovements(routesByDay)
        # Each pairing should consume distinct routes
        consumed = [(p["day1"], p["route1_idx"]) for p in overnightRoutes]
        consumed += [(p["day2"], p["route2_idx"]) for p in overnightRoutes]
        self.assertEqual(len(consumed), len(set(consumed)))

    def test_emptyRoutesProducesNoPairings(self):
        emptyRoutesByDay = {day: [] for day in DAYS}
        overnightRoutes, _ = applyOvernightImprovements(emptyRoutesByDay)
        self.assertEqual(overnightRoutes, [])


class TestOvernightSolver(unittest.TestCase):

    def setUp(self):
        self.solver = OvernightSolver(ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True))
        self.routesByDay, self.overnightRoutes, self.usedRoutes = self.solver.solve(ALL_DF)

    def test_solveReturnsTuple(self):
        result = self.solver.solve(ALL_DF)
        self.assertEqual(len(result), 3)

    def test_routesByDayHasAllDays(self):
        for day in DAYS:
            self.assertIn(day, self.routesByDay)

    def test_usedRoutesHasAllDays(self):
        for day in DAYS:
            self.assertIn(day, self.usedRoutes)

    def test_statsHasRequiredKeys(self):
        stats = self.solver.getStats()
        for key in ["miles", "routes", "feasible", "runtime_s", "overnight_pairs"]:
            self.assertIn(key, stats)

    def test_statsMilesPositive(self):
        self.assertGreater(self.solver.getStats()["miles"], 0)

    def test_statsRoutesPositive(self):
        self.assertGreater(self.solver.getStats()["routes"], 0)

    def test_convergenceNone(self):
        self.assertIsNone(self.solver.getConvergence())

    def test_weightHistoryNone(self):
        self.assertIsNone(self.solver.getWeightHistory())

    def test_overnightPairsCountMatchesStats(self):
        stats = self.solver.getStats()
        self.assertEqual(stats["overnight_pairs"], len(self.overnightRoutes))


if __name__ == "__main__":
    unittest.main()
