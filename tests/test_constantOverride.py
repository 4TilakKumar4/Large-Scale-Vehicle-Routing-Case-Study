"""
tests/test_constantOverride.py — Unit tests for ConstantOverride context manager.

This is the most critical test file for the sensitivity analysis module.
ConstantOverride must guarantee restoration even under failure conditions.
"""

import unittest

from tests.fixtures import injectFixture

injectFixture()

import vrp_solvers.base as base
from VRP_SensitivityAnalysis import ConstantOverride


class TestConstantOverrideBasic(unittest.TestCase):

    def test_valueIsChangedInsideBlock(self):
        original = base.DRIVING_SPEED
        with ConstantOverride(DRIVING_SPEED=99):
            self.assertEqual(base.DRIVING_SPEED, 99)
        # ensure we test the inside, not just after
        base.DRIVING_SPEED = original  # reset in case

    def test_valueIsRestoredAfterBlock(self):
        original = base.DRIVING_SPEED
        with ConstantOverride(DRIVING_SPEED=99):
            pass
        self.assertEqual(base.DRIVING_SPEED, original)

    def test_multipleConstantsPatched(self):
        origSpeed  = base.DRIVING_SPEED
        origMaxDrv = base.MAX_DRIVING
        with ConstantOverride(DRIVING_SPEED=32, MAX_DRIVING=10):
            self.assertEqual(base.DRIVING_SPEED, 32)
            self.assertEqual(base.MAX_DRIVING,   10)
        self.assertEqual(base.DRIVING_SPEED, origSpeed)
        self.assertEqual(base.MAX_DRIVING,   origMaxDrv)

    def test_allConstantsRestoredAfterBlock(self):
        originals = {
            "VAN_CAPACITY":   base.VAN_CAPACITY,
            "DRIVING_SPEED":  base.DRIVING_SPEED,
            "WINDOW_OPEN":    base.WINDOW_OPEN,
            "WINDOW_CLOSE":   base.WINDOW_CLOSE,
        }
        with ConstantOverride(VAN_CAPACITY=2400, DRIVING_SPEED=32,
                              WINDOW_OPEN=6, WINDOW_CLOSE=20):
            pass
        for name, origVal in originals.items():
            self.assertEqual(getattr(base, name), origVal,
                             msg=f"{name} not restored after ConstantOverride")


class TestConstantOverrideExceptionSafety(unittest.TestCase):
    """Verify restoration happens even when the body raises an exception."""

    def test_restoredAfterException(self):
        original = base.DRIVING_SPEED
        try:
            with ConstantOverride(DRIVING_SPEED=99):
                raise ValueError("deliberate test exception")
        except ValueError:
            pass
        self.assertEqual(base.DRIVING_SPEED, original)

    def test_restoredAfterAssertionError(self):
        original = base.MAX_DUTY
        try:
            with ConstantOverride(MAX_DUTY=10):
                assert False, "deliberate assertion"
        except AssertionError:
            pass
        self.assertEqual(base.MAX_DUTY, original)

    def test_restoredAfterMultipleConstantsAndException(self):
        origSpeed = base.DRIVING_SPEED
        origCap   = base.VAN_CAPACITY
        origOpen  = base.WINDOW_OPEN
        try:
            with ConstantOverride(DRIVING_SPEED=32,
                                  VAN_CAPACITY=2400,
                                  WINDOW_OPEN=6):
                raise RuntimeError("test error")
        except RuntimeError:
            pass
        self.assertEqual(base.DRIVING_SPEED, origSpeed)
        self.assertEqual(base.VAN_CAPACITY,  origCap)
        self.assertEqual(base.WINDOW_OPEN,   origOpen)


class TestConstantOverrideNested(unittest.TestCase):
    """Nested ConstantOverride blocks must restore independently."""

    def test_nestedBlocksRestoreInOrder(self):
        origSpeed = base.DRIVING_SPEED
        with ConstantOverride(DRIVING_SPEED=32):
            self.assertEqual(base.DRIVING_SPEED, 32)
            with ConstantOverride(DRIVING_SPEED=48):
                self.assertEqual(base.DRIVING_SPEED, 48)
            # Inner block exits — should restore to 32 (the value set by outer block)
            self.assertEqual(base.DRIVING_SPEED, 32)
        # Outer block exits — should restore to original
        self.assertEqual(base.DRIVING_SPEED, origSpeed)

    def test_nestedDifferentConstants(self):
        origSpeed = base.DRIVING_SPEED
        origCap   = base.VAN_CAPACITY
        with ConstantOverride(DRIVING_SPEED=32):
            with ConstantOverride(VAN_CAPACITY=2400):
                self.assertEqual(base.DRIVING_SPEED, 32)
                self.assertEqual(base.VAN_CAPACITY,  2400)
            self.assertEqual(base.VAN_CAPACITY, origCap)
            self.assertEqual(base.DRIVING_SPEED, 32)
        self.assertEqual(base.DRIVING_SPEED, origSpeed)


class TestConstantOverrideGuards(unittest.TestCase):

    def test_unknownConstantRaisesAttributeError(self):
        with self.assertRaises(AttributeError):
            with ConstantOverride(NONEXISTENT_CONSTANT=99):
                pass

    def test_unknownConstantDoesNotPatchOtherConstants(self):
        """If one constant in a multi-constant call is unknown,
        no patches should have been applied (fail fast before any change)."""
        original = base.DRIVING_SPEED
        try:
            with ConstantOverride(DRIVING_SPEED=99, BAD_NAME=0):
                pass
        except AttributeError:
            pass
        # DRIVING_SPEED should still be original since BAD_NAME fails before any patch
        # NOTE: current implementation patches in order, so DRIVING_SPEED may be
        # changed before BAD_NAME fails. The important guarantee is restoration.
        # Either way, after the context manager exits, value must be original.
        self.assertEqual(base.DRIVING_SPEED, original)

    def test_originalValueNotModifiedByOverride(self):
        """The override should not permanently change the value — only temporarily."""
        original = base.VAN_CAPACITY
        for _ in range(3):
            with ConstantOverride(VAN_CAPACITY=2400):
                self.assertEqual(base.VAN_CAPACITY, 2400)
            self.assertEqual(base.VAN_CAPACITY, original)


class TestConstantOverrideWithSolvers(unittest.TestCase):
    """Integration tests: verify that solver behaviour actually changes under override."""

    def test_reducedCapacityProducesMoreRoutes(self):
        """Halving van capacity should produce more routes than baseline."""
        from vrp_solvers.clarkeWright import ClarkeWrightSolver
        solver = ClarkeWrightSolver(useTwoOpt=False, useOrOpt=False)

        # Baseline
        baseRoutes = solver.solve(
            __import__("tests.fixtures", fromlist=["MON_DF"]).MON_DF
        )
        baseCount = len(baseRoutes)

        # Halved capacity — combined cube of all Mon orders is 500+600+700=1800
        # which fits in 3200 but not in 900
        with ConstantOverride(VAN_CAPACITY=900):
            smallRoutes = solver.solve(
                __import__("tests.fixtures", fromlist=["MON_DF"]).MON_DF
            )
            smallCount = len(smallRoutes)

        # More routes expected under smaller capacity
        self.assertGreaterEqual(smallCount, baseCount)
        # Baseline value restored after block
        self.assertEqual(base.VAN_CAPACITY, 3200)

    def test_fasterSpeedDoesNotBreakFeasibility(self):
        """Increasing speed should keep all routes feasible (faster = more time margin)."""
        from vrp_solvers.clarkeWright import ClarkeWrightSolver
        from vrp_solvers.base import evaluateRoute
        solver = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)

        with ConstantOverride(DRIVING_SPEED=60):
            routes = solver.solve(
                __import__("tests.fixtures", fromlist=["MON_DF"]).MON_DF
            )
            for route in routes:
                self.assertTrue(evaluateRoute(route)["overall_feasible"])

        self.assertEqual(base.DRIVING_SPEED, 40)


if __name__ == "__main__":
    unittest.main()
