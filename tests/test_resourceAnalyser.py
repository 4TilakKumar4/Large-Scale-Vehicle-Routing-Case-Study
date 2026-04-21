"""
tests/test_resourceAnalyser.py — Unit tests for vrp_solvers/resourceAnalyser.py.
"""

import unittest

from tests.fixtures import injectFixture, MON_ORDERS, TUE_ORDERS, ALL_DF, DEPOT

injectFixture()

from vrp_solvers.base import DAYS, evaluateRoute
from vrp_solvers.clarkeWright import ClarkeWrightSolver
from vrp_solvers.resourceAnalyser import ResourceAnalyser


def buildRoutesByDay(orders=ALL_DF):
    """Helper: run CW on all days and return routesByDay dict."""
    solver      = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)
    routesByDay = {}
    for day in DAYS:
        df = orders[orders["DayOfWeek"] == day].copy()
        routesByDay[day] = solver.solve(df)
    return routesByDay


class TestResourceAnalyserGuards(unittest.TestCase):

    def test_reportBeforeAnalyseRaisesRuntimeError(self):
        analyser = ResourceAnalyser(buildRoutesByDay())
        with self.assertRaises(RuntimeError):
            analyser.getReport()

    def test_printReportBeforeAnalyseRaisesRuntimeError(self):
        analyser = ResourceAnalyser(buildRoutesByDay())
        with self.assertRaises(RuntimeError):
            analyser.printReport()


class TestResourceAnalyserTrucks(unittest.TestCase):

    def setUp(self):
        self.routesByDay = buildRoutesByDay()
        self.analyser    = ResourceAnalyser(self.routesByDay)
        self.analyser.analyse()
        self.report      = self.analyser.getReport()

    def test_trucksByDayHasAllDays(self):
        for day in DAYS:
            self.assertIn(day, self.report["trucks_by_day"])

    def test_truckCountMatchesRouteCount(self):
        # Trucks on a given day = number of routes on that day
        for day in DAYS:
            expected = len(self.routesByDay[day])
            self.assertEqual(self.report["trucks_by_day"][day], expected)

    def test_peakTrucksIsMaxAcrossDays(self):
        maxTrucks = max(self.report["trucks_by_day"].values())
        self.assertEqual(self.report["min_trucks_peak"], maxTrucks)

    def test_peakTrucksPositive(self):
        self.assertGreater(self.report["min_trucks_peak"], 0)


class TestResourceAnalyserDrivers(unittest.TestCase):

    def setUp(self):
        self.routesByDay = buildRoutesByDay()
        self.analyser    = ResourceAnalyser(self.routesByDay)
        self.analyser.analyse()
        self.report      = self.analyser.getReport()

    def test_minDriversPositive(self):
        self.assertGreater(self.report["min_drivers"], 0)

    def test_minDriversAtMostTotalRoutes(self):
        # Can never need more drivers than there are routes
        self.assertLessEqual(
            self.report["min_drivers"],
            self.report["total_routes"]
        )

    def test_minDriversAtLeastPeakDayRoutes(self):
        # Need at least as many drivers as the busiest single day requires simultaneously
        self.assertGreaterEqual(
            self.report["min_drivers"],
            self.report["min_trucks_peak"]
        )

    def test_reportTotalRoutesCorrect(self):
        expected = sum(len(r) for r in self.routesByDay.values())
        self.assertEqual(self.report["total_routes"], expected)

    def test_overnightPairsZeroWithNoOvernight(self):
        self.assertEqual(self.report["overnight_pairs"], 0)


class TestResourceAnalyserOvernightPairings(unittest.TestCase):

    def test_overnightPairingReducesDriverCount(self):
        # Build two separate routes and compare driver count with and without pairing
        routesByDay = buildRoutesByDay()

        # Without overnight pairing
        analyserNormal = ResourceAnalyser(routesByDay)
        analyserNormal.analyse()
        driversNormal = analyserNormal.getReport()["min_drivers"]

        # With a synthetic overnight pairing consuming one Mon route and one Tue route
        monRoutes = routesByDay.get("Mon", [])
        tueRoutes = routesByDay.get("Tue", [])

        if monRoutes and tueRoutes:
            fakePairing = [{
                "day1":       "Mon",
                "day2":       "Tue",
                "route1_idx": 0,
                "route2_idx": 0,
                "route1":     monRoutes[0],
                "route2":     tueRoutes[0],
                "results":    {"overall_feasible": True},
                "savings":    10,
            }]
            analyserOvernight = ResourceAnalyser(routesByDay, overnightPairings=fakePairing)
            analyserOvernight.analyse()
            driversOvernight = analyserOvernight.getReport()["min_drivers"]

            # The overnight pairing pre-commits one driver to two routes,
            # so drivers should be ≤ without-overnight count
            self.assertLessEqual(driversOvernight, driversNormal)

    def test_overnightPairsCountReflectedInReport(self):
        routesByDay = buildRoutesByDay()
        monRoutes   = routesByDay.get("Mon", [])
        tueRoutes   = routesByDay.get("Tue", [])

        if monRoutes and tueRoutes:
            fakePairings = [
                {
                    "day1": "Mon", "day2": "Tue",
                    "route1_idx": 0, "route2_idx": 0,
                    "route1": monRoutes[0], "route2": tueRoutes[0],
                    "results": {"overall_feasible": True}, "savings": 10,
                }
            ]
            analyser = ResourceAnalyser(routesByDay, overnightPairings=fakePairings)
            analyser.analyse()
            self.assertEqual(analyser.getReport()["overnight_pairs"], 1)


class TestResourceAnalyserDriverChains(unittest.TestCase):

    def test_driverChainsNotEmpty(self):
        analyser = ResourceAnalyser(buildRoutesByDay())
        analyser.analyse()
        # _driverChains is populated after analyse()
        self.assertIsNotNone(analyser._driverChains)
        self.assertGreater(len(analyser._driverChains), 0)

    def test_eachChainIsListOfTuples(self):
        analyser = ResourceAnalyser(buildRoutesByDay())
        analyser.analyse()
        for chain in analyser._driverChains:
            self.assertIsInstance(chain, list)
            for node in chain:
                self.assertIsInstance(node, tuple)
                self.assertEqual(len(node), 2)




class TestResourceAnalyserWorkloadMetrics(unittest.TestCase):
    """Tests for the driver workload metrics added to getReport() and toDataFrame()."""

    def setUp(self):
        self.routesByDay = buildRoutesByDay()
        self.analyser    = ResourceAnalyser(self.routesByDay)
        self.analyser.analyse()
        self.report      = self.analyser.getReport()

    def test_reportHasAvgWeeklyDutyHrs(self):
        self.assertIn("avg_weekly_duty_hrs", self.report)

    def test_reportHasMaxWeeklyDutyHrs(self):
        self.assertIn("max_weekly_duty_hrs", self.report)

    def test_avgWeeklyDutyHrsIsPositive(self):
        self.assertGreater(self.report["avg_weekly_duty_hrs"], 0.0)

    def test_maxWeeklyDutyHrsIsPositive(self):
        self.assertGreater(self.report["max_weekly_duty_hrs"], 0.0)

    def test_maxWeeklyDutyHrsGeqAvg(self):
        self.assertGreaterEqual(
            self.report["max_weekly_duty_hrs"],
            self.report["avg_weekly_duty_hrs"]
        )

    def test_avgWeeklyDutyHrsIsReasonable(self):
        # Each driver works at most 5 days × 14h = 70h; must be positive
        self.assertLessEqual(self.report["avg_weekly_duty_hrs"], 70.0)

    def test_maxWeeklyDutyHrsIsReasonable(self):
        self.assertLessEqual(self.report["max_weekly_duty_hrs"], 70.0)


class TestResourceAnalyserChainsDataFrame(unittest.TestCase):
    """Tests for the new columns in the chainsDF returned by toDataFrame()."""

    def setUp(self):
        analyser = ResourceAnalyser(buildRoutesByDay())
        analyser.analyse()
        _, self.chainsDF = analyser.toDataFrame()

    def test_chainsDFHasWeeklyDutyHrs(self):
        self.assertIn("weekly_duty_hrs", self.chainsDF.columns)

    def test_chainsDFHasAvgDutyPerDay(self):
        self.assertIn("avg_duty_per_day", self.chainsDF.columns)

    def test_weeklyDutyHrsPositive(self):
        self.assertTrue((self.chainsDF["weekly_duty_hrs"] > 0).all())

    def test_avgDutyPerDayPositive(self):
        self.assertTrue((self.chainsDF["avg_duty_per_day"] > 0).all())

    def test_avgDutyPerDayLeqWeeklyDuty(self):
        # avg per day <= weekly total for all drivers
        self.assertTrue(
            (self.chainsDF["avg_duty_per_day"] <= self.chainsDF["weekly_duty_hrs"]).all()
        )

    def test_avgDutyPerDayConsistentWithDaysWorked(self):
        # avg_duty_per_day == weekly_duty_hrs / num_days_worked for each row
        for _, row in self.chainsDF.iterrows():
            if row["num_days_worked"] > 0:
                expected = round(row["weekly_duty_hrs"] / row["num_days_worked"], 2)
                self.assertAlmostEqual(row["avg_duty_per_day"], expected, places=1)

    def test_summaryDFHasWorkloadColumns(self):
        analyser = ResourceAnalyser(buildRoutesByDay())
        analyser.analyse()
        summaryDF, _ = analyser.toDataFrame()
        self.assertIn("avg_weekly_duty_hrs", summaryDF.columns)
        self.assertIn("max_weekly_duty_hrs", summaryDF.columns)

if __name__ == "__main__":
    unittest.main()
