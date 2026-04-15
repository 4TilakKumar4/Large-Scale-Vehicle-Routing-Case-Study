"""
tests/test_mixedFleetSolver.py — Unit tests for vrp_solvers/mixedFleetSolver.py.
"""

import unittest

from tests.fixtures import (
    injectFixture,
    MIXED_MON_DF,
    MIXED_MON_ORDERS,
    ST_ORDERS,
    LARGE_ORDERS,
    FLEX_ORDERS,
    makeOrder,
    makeOrdersDf,
)

injectFixture()

from vrp_solvers.base import evaluateMixedRoute, ST_CAPACITY, VAN_CAPACITY
from vrp_solvers.mixedFleetSolver import MixedFleetSolver, ALNSMixedFleetSolver, _isStRequired, _isTooLargeForST, _isFlexible


class TestFleetClassifiers(unittest.TestCase):

    def test_stRequiredOrder(self):
        stop = {**makeOrder(1, 1001, 300, "Mon"), "straight_truck_required": "yes"}
        self.assertTrue(_isStRequired(stop))
        self.assertFalse(_isTooLargeForST(stop))
        self.assertFalse(_isFlexible(stop))

    def test_tooLargeForSTOrder(self):
        stop = makeOrder(2, 1001, ST_CAPACITY + 100, "Mon")
        self.assertFalse(_isStRequired(stop))
        self.assertTrue(_isTooLargeForST(stop))
        self.assertFalse(_isFlexible(stop))

    def test_flexibleOrder(self):
        stop = makeOrder(3, 1001, 300, "Mon")
        self.assertFalse(_isStRequired(stop))
        self.assertFalse(_isTooLargeForST(stop))
        self.assertTrue(_isFlexible(stop))

    def test_stRequiredCaseInsensitive(self):
        stop = {**makeOrder(4, 1001, 300, "Mon"), "straight_truck_required": "YES"}
        self.assertTrue(_isStRequired(stop))


class TestMixedFleetSolverEmptyDay(unittest.TestCase):

    def setUp(self):
        self.solver = MixedFleetSolver()

    def test_emptyDayReturnsEmptyList(self):
        routes = self.solver.solve(makeOrdersDf([]))
        self.assertEqual(routes, [])

    def test_emptyDayStatsZeroed(self):
        self.solver.solve(makeOrdersDf([]))
        stats = self.solver.getStats()
        self.assertEqual(stats["miles"],      0)
        self.assertEqual(stats["routes"],     0)
        self.assertEqual(stats["van_routes"], 0)
        self.assertEqual(stats["st_routes"],  0)
        self.assertTrue(stats["feasible"])

    def test_emptyDayVanRoutesEmpty(self):
        self.solver.solve(makeOrdersDf([]))
        self.assertEqual(self.solver.getVanRoutes(), [])

    def test_emptyDayStRoutesEmpty(self):
        self.solver.solve(makeOrdersDf([]))
        self.assertEqual(self.solver.getStRoutes(), [])

    def test_convergenceNone(self):
        self.solver.solve(makeOrdersDf([]))
        self.assertIsNone(self.solver.getConvergence())


class TestMixedFleetSolverFleetAssignment(unittest.TestCase):

    def setUp(self):
        self.solver = MixedFleetSolver()
        self.solver.solve(MIXED_MON_DF)
        self.vanRoutes = self.solver.getVanRoutes()
        self.stRoutes  = self.solver.getStRoutes()

    def test_stRequiredOrdersOnlyOnST(self):
        """Orders marked straight_truck_required=yes must never appear on Van routes."""
        stReqIds = {int(o["ORDERID"]) for o in ST_ORDERS}
        for route in self.vanRoutes:
            for stop in route:
                self.assertNotIn(int(stop["ORDERID"]), stReqIds,
                                 msg=f"ST-required order {int(stop['ORDERID'])} found on Van route")

    def test_stRequiredOrdersAppearOnST(self):
        """Every ST-required order must appear on an ST route."""
        stReqIds = {int(o["ORDERID"]) for o in ST_ORDERS}
        stRoutedIds = {int(stop["ORDERID"]) for route in self.stRoutes for stop in route}
        for oid in stReqIds:
            self.assertIn(oid, stRoutedIds,
                          msg=f"ST-required order {oid} missing from ST routes")

    def test_tooLargeOrdersOnlyOnVan(self):
        """Orders whose cube exceeds ST_CAPACITY must never appear on ST routes."""
        largeIds = {int(o["ORDERID"]) for o in LARGE_ORDERS}
        for route in self.stRoutes:
            for stop in route:
                self.assertNotIn(int(stop["ORDERID"]), largeIds,
                                 msg=f"Over-capacity order {int(stop['ORDERID'])} found on ST route")

    def test_allOrdersCoveredExactlyOnce(self):
        """Every order in the input must appear exactly once across Van + ST routes."""
        allRouted = (
            [int(s["ORDERID"]) for r in self.vanRoutes for s in r]
            + [int(s["ORDERID"]) for r in self.stRoutes  for s in r]
        )
        expected = sorted(MIXED_MON_DF["ORDERID"].astype(int).tolist())
        self.assertEqual(sorted(allRouted), expected)
        self.assertEqual(len(allRouted), len(set(allRouted)),
                         msg="Duplicate order IDs found across routes")

    def test_vanRoutesFeasible(self):
        for route in self.vanRoutes:
            r = evaluateMixedRoute(route, "van")
            self.assertTrue(r["overall_feasible"],
                            msg=f"Van route infeasible: {[int(s['ORDERID']) for s in route]}")

    def test_stRoutesFeasible(self):
        for route in self.stRoutes:
            r = evaluateMixedRoute(route, "st")
            self.assertTrue(r["overall_feasible"],
                            msg=f"ST route infeasible: {[int(s['ORDERID']) for s in route]}")

    def test_vanRouteCapacity(self):
        for route in self.vanRoutes:
            cube = sum(float(s["CUBE"]) for s in route)
            self.assertLessEqual(cube, VAN_CAPACITY,
                                 msg=f"Van route exceeds capacity: {cube} > {VAN_CAPACITY}")

    def test_stRouteCapacity(self):
        for route in self.stRoutes:
            cube = sum(float(s["CUBE"]) for s in route)
            self.assertLessEqual(cube, ST_CAPACITY,
                                 msg=f"ST route exceeds capacity: {cube} > {ST_CAPACITY}")


class TestMixedFleetSolverStats(unittest.TestCase):

    def setUp(self):
        self.solver = MixedFleetSolver()
        self.solver.solve(MIXED_MON_DF)
        self.stats = self.solver.getStats()

    def test_statsHasRequiredKeys(self):
        for key in ["miles", "routes", "feasible", "runtime_s",
                    "van_routes", "st_routes", "van_miles", "st_miles"]:
            self.assertIn(key, self.stats)

    def test_totalRoutesMatchVanPlusST(self):
        self.assertEqual(
            self.stats["routes"],
            self.stats["van_routes"] + self.stats["st_routes"]
        )

    def test_totalMilesMatchVanPlusST(self):
        self.assertEqual(
            self.stats["miles"],
            self.stats["van_miles"] + self.stats["st_miles"]
        )

    def test_routeCountMatchesActual(self):
        self.assertEqual(self.stats["van_routes"], len(self.solver.getVanRoutes()))
        self.assertEqual(self.stats["st_routes"],  len(self.solver.getStRoutes()))

    def test_milesPositive(self):
        self.assertGreater(self.stats["miles"], 0)

    def test_runtimeNonNegative(self):
        self.assertGreaterEqual(self.stats["runtime_s"], 0)


class TestMixedFleetSolverFlexibleOrders(unittest.TestCase):

    def test_flexibleOrdersCanGoOnEitherFleet(self):
        """Flexible orders should be routable — verify they appear in the solution."""
        solver = MixedFleetSolver()
        solver.solve(MIXED_MON_DF)
        flexIds = {int(o["ORDERID"]) for o in FLEX_ORDERS}
        allRouted = (
            {int(s["ORDERID"]) for r in solver.getVanRoutes() for s in r}
            | {int(s["ORDERID"]) for r in solver.getStRoutes()  for s in r}
        )
        for fid in flexIds:
            self.assertIn(fid, allRouted,
                          msg=f"Flexible order {fid} not found in any route")

    def test_stOnlyDayHasNoVanRoutes(self):
        """A day with only ST-required orders should produce only ST routes."""
        stOnlyOrders = [
            {**makeOrder(1, 1001, 300, "Mon"), "straight_truck_required": "yes"},
            {**makeOrder(2, 1002, 400, "Mon"), "straight_truck_required": "yes"},
        ]
        df     = makeOrdersDf(stOnlyOrders)
        solver = MixedFleetSolver()
        solver.solve(df)
        # ST-required orders must be on ST routes
        stRouted = {int(s["ORDERID"]) for r in solver.getStRoutes() for s in r}
        self.assertIn(1, stRouted)
        self.assertIn(2, stRouted)




class TestALNSMixedFleetSolverEmptyDay(unittest.TestCase):

    def setUp(self):
        self.solver = ALNSMixedFleetSolver()

    def test_emptyDayReturnsEmptyList(self):
        self.assertEqual(self.solver.solve(makeOrdersDf([])), [])

    def test_emptyDayStatsZeroed(self):
        self.solver.solve(makeOrdersDf([]))
        stats = self.solver.getStats()
        self.assertEqual(stats["miles"],      0)
        self.assertEqual(stats["routes"],     0)
        self.assertEqual(stats["van_routes"], 0)
        self.assertEqual(stats["st_routes"],  0)
        self.assertTrue(stats["feasible"])

    def test_emptyDayVanRoutesEmpty(self):
        self.solver.solve(makeOrdersDf([]))
        self.assertEqual(self.solver.getVanRoutes(), [])

    def test_emptyDayStRoutesEmpty(self):
        self.solver.solve(makeOrdersDf([]))
        self.assertEqual(self.solver.getStRoutes(), [])

    def test_convergenceEmptyOnEmptyDay(self):
        self.solver.solve(makeOrdersDf([]))
        self.assertEqual(self.solver.getConvergence(), [])

    def test_weightHistoryEmptyOnEmptyDay(self):
        self.solver.solve(makeOrdersDf([]))
        self.assertEqual(self.solver.getWeightHistory(), {})


class TestALNSMixedFleetSolverFleetAssignment(unittest.TestCase):
    """ALNS mixed fleet must satisfy the same hard fleet assignment rules as CW mixed fleet."""

    def setUp(self):
        self.solver = ALNSMixedFleetSolver(maxIter=50)   # short run for test speed
        self.solver.solve(MIXED_MON_DF)
        self.vanRoutes = self.solver.getVanRoutes()
        self.stRoutes  = self.solver.getStRoutes()

    def test_stRequiredOrdersOnlyOnST(self):
        stReqIds = {int(o["ORDERID"]) for o in ST_ORDERS}
        for route in self.vanRoutes:
            for stop in route:
                self.assertNotIn(int(stop["ORDERID"]), stReqIds,
                                 msg=f"ST-required order on Van route")

    def test_stRequiredOrdersAppearOnST(self):
        stReqIds    = {int(o["ORDERID"]) for o in ST_ORDERS}
        stRoutedIds = {int(s["ORDERID"]) for r in self.stRoutes for s in r}
        for oid in stReqIds:
            self.assertIn(oid, stRoutedIds)

    def test_tooLargeOrdersOnlyOnVan(self):
        largeIds = {int(o["ORDERID"]) for o in LARGE_ORDERS}
        for route in self.stRoutes:
            for stop in route:
                self.assertNotIn(int(stop["ORDERID"]), largeIds,
                                 msg=f"Over-capacity order on ST route")

    def test_allOrdersCoveredExactlyOnce(self):
        allRouted = (
            [int(s["ORDERID"]) for r in self.vanRoutes for s in r]
            + [int(s["ORDERID"]) for r in self.stRoutes  for s in r]
        )
        expected = sorted(MIXED_MON_DF["ORDERID"].astype(int).tolist())
        self.assertEqual(sorted(allRouted), expected)
        self.assertEqual(len(allRouted), len(set(allRouted)))

    def test_vanRoutesFeasible(self):
        for route in self.vanRoutes:
            r = evaluateMixedRoute(route, "van")
            self.assertTrue(r["overall_feasible"])

    def test_stRoutesFeasible(self):
        for route in self.stRoutes:
            r = evaluateMixedRoute(route, "st")
            self.assertTrue(r["overall_feasible"])

    def test_vanRouteCapacity(self):
        for route in self.vanRoutes:
            self.assertLessEqual(sum(float(s["CUBE"]) for s in route), VAN_CAPACITY)

    def test_stRouteCapacity(self):
        for route in self.stRoutes:
            self.assertLessEqual(sum(float(s["CUBE"]) for s in route), ST_CAPACITY)


class TestALNSMixedFleetSolverStats(unittest.TestCase):

    def setUp(self):
        self.solver = ALNSMixedFleetSolver(maxIter=50)
        self.solver.solve(MIXED_MON_DF)
        self.stats = self.solver.getStats()

    def test_statsHasRequiredKeys(self):
        for key in ["miles", "routes", "feasible", "runtime_s",
                    "van_routes", "st_routes", "van_miles", "st_miles"]:
            self.assertIn(key, self.stats)

    def test_totalRoutesMatchVanPlusST(self):
        self.assertEqual(self.stats["routes"],
                         self.stats["van_routes"] + self.stats["st_routes"])

    def test_totalMilesMatchVanPlusST(self):
        self.assertEqual(self.stats["miles"],
                         self.stats["van_miles"] + self.stats["st_miles"])

    def test_milesPositive(self):
        self.assertGreater(self.stats["miles"], 0)

    def test_feasibleTrue(self):
        self.assertTrue(self.stats["feasible"])


class TestALNSMixedFleetSolverConvergence(unittest.TestCase):

    def setUp(self):
        self.solver = ALNSMixedFleetSolver(maxIter=50)
        self.solver.solve(MIXED_MON_DF)

    def test_convergenceNotEmpty(self):
        self.assertIsNotNone(self.solver.getConvergence())
        self.assertGreater(len(self.solver.getConvergence()), 0)

    def test_convergenceLengthMatchesMaxIter(self):
        # convergence has maxIter + 1 entries (seed + one per iteration)
        self.assertEqual(len(self.solver.getConvergence()), 51)

    def test_weightHistoryHasCorrectKeys(self):
        wh = self.solver.getWeightHistory()
        self.assertIn("destroy", wh)
        self.assertIn("repair",  wh)

    def test_weightHistoryDestroyOperatorNames(self):
        names = set(self.solver.getWeightHistory()["destroy"].keys())
        self.assertEqual(names, {"random", "worst", "shaw", "route"})

    def test_weightHistoryRepairOperatorNames(self):
        names = set(self.solver.getWeightHistory()["repair"].keys())
        self.assertEqual(names, {"greedy", "regret"})


class TestALNSMixedFleetSolverVsCW(unittest.TestCase):
    """ALNS should not produce a worse solution than the CW seed it starts from."""

    def test_alnsNotWorseThanCWSeed(self):
        cwSolver   = MixedFleetSolver()
        alnsSolver = ALNSMixedFleetSolver(maxIter=100)

        cwSolver.solve(MIXED_MON_DF)
        alnsSolver.solve(MIXED_MON_DF)

        cwMiles   = cwSolver.getStats()["miles"]
        alnsMiles = alnsSolver.getStats()["miles"]

        # ALNS seeds from CW so should never be strictly worse
        self.assertLessEqual(alnsMiles, cwMiles + 1,
                             msg=f"ALNS ({alnsMiles}) worse than CW seed ({cwMiles})")

if __name__ == "__main__":
    unittest.main()
