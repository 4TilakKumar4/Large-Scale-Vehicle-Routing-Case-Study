"""
tests/test_costModel.py — Unit tests for vrp_solvers/costModel.py.

Uses hand-crafted result dicts wherever possible — no solver calls needed
for component-level tests. Solver-based tests use the synthetic fixture.
"""

import unittest

from tests.fixtures import injectFixture, MON_DF, makeOrdersDf

injectFixture()

from vrp_solvers.costModel import CostModel


# Synthetic route result dicts — hand-crafted to give predictable values
RESULT_100MI_8H = {
    "total_miles": 100,
    "total_duty":  8.0,
    "total_drive": 2.5,
    "total_cube":  500,
    "overall_feasible": True,
}

RESULT_200MI_10H = {
    "total_miles": 200,
    "total_duty":  10.0,
    "total_drive": 5.0,
    "total_cube":  800,
    "overall_feasible": True,
}

# Overnight result dict — has day1_duty and day2_duty
OVERNIGHT_RESULT = {
    "total_miles":  250,
    "day1_duty":    9.0,
    "day2_duty":    7.0,
    "overall_feasible": True,
    "day1_drive":   6.0,
    "day2_drive":   4.0,
}


class TestCostModelDefaults(unittest.TestCase):

    def setUp(self):
        self.cm = CostModel()

    def test_defaultRatesCorrect(self):
        self.assertEqual(self.cm.cost_per_mile_van,    0.725)
        self.assertEqual(self.cm.cost_per_mile_st,     0.820)
        self.assertEqual(self.cm.driver_hourly_wage,   32.00)
        self.assertEqual(self.cm.overtime_multiplier,  1.50)
        self.assertEqual(self.cm.regular_shift_hours,  8.0)
        self.assertEqual(self.cm.benefits_loading,     0.30)
        self.assertEqual(self.cm.overnight_allowance,  80.00)
        self.assertEqual(self.cm.sleeper_premium_daily, 60.00)
        self.assertEqual(self.cm.day_cab_daily,        185.00)
        self.assertEqual(self.cm.st_trailer_premium_daily, 25.00)
        self.assertEqual(self.cm.insurance_per_day,    56.00)
        self.assertEqual(self.cm.weeks_per_year,       52)


class TestMileageCost(unittest.TestCase):

    def setUp(self):
        self.cm = CostModel()

    def test_vanMileageCost(self):
        self.assertAlmostEqual(self.cm._mileageCost(100, "van"), 72.50)

    def test_stMileageCost(self):
        self.assertAlmostEqual(self.cm._mileageCost(100, "st"), 82.00)

    def test_zeroMiles(self):
        self.assertEqual(self.cm._mileageCost(0, "van"), 0.0)

    def test_customRate(self):
        cm = CostModel(cost_per_mile_van=0.80)
        self.assertAlmostEqual(cm._mileageCost(100, "van"), 80.00)


class TestLabourCost(unittest.TestCase):

    def setUp(self):
        self.cm = CostModel()

    def test_regularShiftOnly(self):
        # 8h × $32 = $256.00 — no overtime
        self.assertAlmostEqual(self.cm._labourCost(8.0), 256.00)

    def test_withOvertime(self):
        # 8h × $32 + 2h × $48 = $256 + $96 = $352.00
        self.assertAlmostEqual(self.cm._labourCost(10.0), 352.00)

    def test_zeroHours(self):
        self.assertEqual(self.cm._labourCost(0.0), 0.0)

    def test_exactlyAtOvertimeThreshold(self):
        self.assertAlmostEqual(self.cm._labourCost(8.0), 256.00)

    def test_oneHourOvertime(self):
        # 8h × $32 + 1h × $48 = $256 + $48 = $304
        self.assertAlmostEqual(self.cm._labourCost(9.0), 304.00)

    def test_customWage(self):
        cm = CostModel(driver_hourly_wage=40.00)
        # 8h × $40 = $320 (no overtime)
        self.assertAlmostEqual(cm._labourCost(8.0), 320.00)


class TestBenefitsCost(unittest.TestCase):

    def setUp(self):
        self.cm = CostModel()

    def test_standardBenefits(self):
        # 30% of $256 = $76.80
        self.assertAlmostEqual(self.cm._benefitsCost(256.00), 76.80)

    def test_zeroWages(self):
        self.assertEqual(self.cm._benefitsCost(0.0), 0.0)

    def test_customBenefitsLoading(self):
        cm = CostModel(benefits_loading=0.25)
        self.assertAlmostEqual(cm._benefitsCost(256.00), 64.00)


class TestEquipmentCost(unittest.TestCase):

    def setUp(self):
        self.cm = CostModel()

    def test_vanDayCab(self):
        self.assertEqual(self.cm._equipmentCost("van", False), 185.00)

    def test_vanSleeper(self):
        # 185 + 60 sleeper = 245
        self.assertEqual(self.cm._equipmentCost("van", True), 245.00)

    def test_stDayCab(self):
        # 185 + 25 ST trailer premium = 210
        self.assertEqual(self.cm._equipmentCost("st", False), 210.00)

    def test_stSleeper(self):
        # 185 + 25 + 60 = 270
        self.assertEqual(self.cm._equipmentCost("st", True), 270.00)

    def test_customDayCabRate(self):
        cm = CostModel(day_cab_daily=200.00)
        self.assertEqual(cm._equipmentCost("van", False), 200.00)


class TestRouteCost(unittest.TestCase):

    def setUp(self):
        self.cm = CostModel()

    def test_totalEqualsComponentSum(self):
        result = self.cm.routeCost(RESULT_100MI_8H, "van")
        componentSum = (result["mileage"] + result["labour"] + result["benefits"]
                        + result["equipment"] + result["insurance"] + result["per_diem"])
        self.assertAlmostEqual(result["total"], componentSum, places=2)

    def test_dayCabPerDiemZero(self):
        result = self.cm.routeCost(RESULT_100MI_8H, "van")
        self.assertEqual(result["per_diem"], 0.0)

    def test_mileageComponent(self):
        result = self.cm.routeCost(RESULT_100MI_8H, "van")
        self.assertAlmostEqual(result["mileage"], 72.50)

    def test_labourComponent(self):
        # 8h regular = 8 × 32 = 256
        result = self.cm.routeCost(RESULT_100MI_8H, "van")
        self.assertAlmostEqual(result["labour"], 256.00)

    def test_overtimeTriggered(self):
        # 10h duty = 8h × 32 + 2h × 48 = 352
        result = self.cm.routeCost(RESULT_200MI_10H, "van")
        self.assertAlmostEqual(result["labour"], 352.00)

    def test_benefitsComponent(self):
        # 30% × 256 = 76.80
        result = self.cm.routeCost(RESULT_100MI_8H, "van")
        self.assertAlmostEqual(result["benefits"], 76.80)

    def test_equipmentComponent(self):
        result = self.cm.routeCost(RESULT_100MI_8H, "van")
        self.assertAlmostEqual(result["equipment"], 185.00)

    def test_insuranceComponent(self):
        result = self.cm.routeCost(RESULT_100MI_8H, "van")
        self.assertAlmostEqual(result["insurance"], 56.00)

    def test_stRouteHigherMileageCost(self):
        vanResult = self.cm.routeCost(RESULT_100MI_8H, "van")
        stResult  = self.cm.routeCost(RESULT_100MI_8H, "st")
        self.assertGreater(stResult["mileage"], vanResult["mileage"])

    def test_stRouteHigherEquipmentCost(self):
        vanResult = self.cm.routeCost(RESULT_100MI_8H, "van")
        stResult  = self.cm.routeCost(RESULT_100MI_8H, "st")
        self.assertGreater(stResult["equipment"], vanResult["equipment"])

    def test_sleeperHigherEquipmentCost(self):
        dayCab  = self.cm.routeCost(RESULT_100MI_8H, "van", isSleeper=False)
        sleeper = self.cm.routeCost(RESULT_100MI_8H, "van", isSleeper=True)
        self.assertGreater(sleeper["equipment"], dayCab["equipment"])
        self.assertAlmostEqual(sleeper["equipment"] - dayCab["equipment"], 60.00)


class TestOvernightRouteCost(unittest.TestCase):

    def setUp(self):
        self.cm = CostModel()

    def test_perDiemPresent(self):
        result = self.cm.overnightRouteCost(OVERNIGHT_RESULT)
        self.assertEqual(result["per_diem"], 80.00)

    def test_twoShiftsLabour(self):
        # day1=9h (8h reg + 1h OT) + day2=7h (all regular)
        # day1: 8×32 + 1×48 = 304; day2: 7×32 = 224; total = 528
        result = self.cm.overnightRouteCost(OVERNIGHT_RESULT)
        self.assertAlmostEqual(result["labour"], 528.00)

    def test_equipmentTwoDays(self):
        # Sleeper cab for 2 days: (185 + 60) × 2 = 490
        result = self.cm.overnightRouteCost(OVERNIGHT_RESULT)
        self.assertAlmostEqual(result["equipment"], 490.00)

    def test_insuranceTwoDays(self):
        result = self.cm.overnightRouteCost(OVERNIGHT_RESULT)
        self.assertAlmostEqual(result["insurance"], 112.00)  # 56 × 2

    def test_totalEqualsComponentSum(self):
        result = self.cm.overnightRouteCost(OVERNIGHT_RESULT)
        componentSum = (result["mileage"] + result["labour"] + result["benefits"]
                        + result["equipment"] + result["insurance"] + result["per_diem"])
        self.assertAlmostEqual(result["total"], componentSum, places=2)


class TestWeeklyCost(unittest.TestCase):

    def setUp(self):
        self.cm = CostModel()

    def test_emptyRoutesReturnsZeros(self):
        from vrp_solvers.base import DAYS
        emptyRoutesByDay = {day: [] for day in DAYS}
        result = self.cm.weeklyCost(emptyRoutesByDay)
        self.assertEqual(result["total"], 0.0)
        for k in ["mileage", "labour", "benefits", "equipment", "insurance", "per_diem"]:
            self.assertEqual(result[k], 0.0)

    def test_totalEqualsComponentSum(self):
        from vrp_solvers.clarkeWright import ClarkeWrightSolver
        from vrp_solvers.base import DAYS
        solver      = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)
        routesByDay = {}
        for day in DAYS:
            df = MON_DF.copy() if day == "Mon" else makeOrdersDf([])
            df["DayOfWeek"] = day
            routesByDay[day] = solver.solve(df)

        result = self.cm.weeklyCost(routesByDay)
        componentSum = sum(result[k] for k in
                           ["mileage", "labour", "benefits", "equipment", "insurance", "per_diem"])
        self.assertAlmostEqual(result["total"], componentSum, places=2)


class TestAnnualCost(unittest.TestCase):

    def setUp(self):
        self.cm = CostModel()

    def test_annualisesCorrectly(self):
        self.assertAlmostEqual(self.cm.annualCost(1000.00), 52000.00)

    def test_zeroWeekly(self):
        self.assertEqual(self.cm.annualCost(0.0), 0.0)

    def test_customWeeksPerYear(self):
        cm = CostModel(weeks_per_year=50)
        self.assertAlmostEqual(cm.annualCost(1000.00), 50000.00)


class TestCustomRateOverrides(unittest.TestCase):

    def test_higherWageRaisesLabour(self):
        cmLow  = CostModel(driver_hourly_wage=28.00)
        cmHigh = CostModel(driver_hourly_wage=40.00)
        self.assertGreater(
            cmHigh.routeCost(RESULT_100MI_8H, "van")["labour"],
            cmLow.routeCost(RESULT_100MI_8H, "van")["labour"],
        )

    def test_higherMileageRateRaisesMileage(self):
        cmLow  = CostModel(cost_per_mile_van=0.65)
        cmHigh = CostModel(cost_per_mile_van=0.85)
        self.assertGreater(
            cmHigh.routeCost(RESULT_100MI_8H, "van")["mileage"],
            cmLow.routeCost(RESULT_100MI_8H, "van")["mileage"],
        )

    def test_higherInsuranceRaisesTotal(self):
        cmLow  = CostModel(insurance_per_day=45.00)
        cmHigh = CostModel(insurance_per_day=75.00)
        self.assertGreater(
            cmHigh.routeCost(RESULT_100MI_8H, "van")["total"],
            cmLow.routeCost(RESULT_100MI_8H, "van")["total"],
        )




class TestOvernightSummary(unittest.TestCase):
    """Tests for CostModel.overnightSummary() — overnight cost vs savings framing."""

    def setUp(self):
        from vrp_solvers.clarkeWright import ClarkeWrightSolver
        from vrp_solvers.overnightSolver import OvernightSolver, applyOvernightImprovements
        from vrp_solvers.base import DAYS

        self.cm = CostModel()

        solver = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)
        self.routesByDay = {}
        for day in DAYS:
            df = MON_DF.copy() if day == "Mon" else makeOrdersDf([])
            df["DayOfWeek"] = day
            self.routesByDay[day] = solver.solve(df)

        # Build a minimal fake overnight pairing so the summary has something to compute
        # Use Mon routes as both day1 and day2 to keep the test self-contained
        monRoutes = self.routesByDay.get("Mon", [])
        if len(monRoutes) >= 2:
            from vrp_solvers.base import evaluateRoute
            r1 = monRoutes[0]
            r2 = monRoutes[1]
            res1 = evaluateRoute(r1)
            res2 = evaluateRoute(r2)
            self.overnightPairings = [{
                "day1": "Mon", "route1_idx": 0,
                "day2": "Tue", "route2_idx": 0,
                "route1": r1, "route2": r2,
                "savings": 5,
                "results": {
                    "total_miles": res1["total_miles"] + res2["total_miles"] - 5,
                    "day1_duty":   res1["total_duty"],
                    "day2_duty":   res2["total_duty"],
                    "overall_feasible": True,
                },
            }]
        else:
            self.overnightPairings = []

        self.summary = self.cm.overnightSummary(self.routesByDay, self.overnightPairings)

    def test_summaryHasRequiredKeys(self):
        for key in ["overnight_pairs", "drivers_in_sleeper", "miles_saved_weekly",
                    "annual_miles_saved", "annual_mileage_saving", "per_diem_annual",
                    "sleeper_annual", "overnight_cost_annual", "net_annual_saving"]:
            self.assertIn(key, self.summary)

    def test_overnightPairsMatchInput(self):
        self.assertEqual(self.summary["overnight_pairs"], len(self.overnightPairings))

    def test_driversInSleeperMatchesPairs(self):
        self.assertEqual(self.summary["drivers_in_sleeper"],
                         self.summary["overnight_pairs"])

    def test_perDiemAnnualFormula(self):
        expected = round(
            self.cm.overnight_allowance * len(self.overnightPairings) * self.cm.weeks_per_year,
            2
        )
        self.assertAlmostEqual(self.summary["per_diem_annual"], expected, places=2)

    def test_sleeperAnnualFormula(self):
        expected = round(
            self.cm.sleeper_premium_daily * 2 * len(self.overnightPairings) * self.cm.weeks_per_year,
            2
        )
        self.assertAlmostEqual(self.summary["sleeper_annual"], expected, places=2)

    def test_overnightCostIsPerDiemPlusSleeper(self):
        expected = round(self.summary["per_diem_annual"] + self.summary["sleeper_annual"], 2)
        self.assertAlmostEqual(self.summary["overnight_cost_annual"], expected, places=2)

    def test_netSavingIsMillageSavingMinusOvernightCost(self):
        expected = round(
            self.summary["annual_mileage_saving"] - self.summary["overnight_cost_annual"], 2
        )
        self.assertAlmostEqual(self.summary["net_annual_saving"], expected, places=2)

    def test_emptyPairingsReturnsZeroCosts(self):
        summary = self.cm.overnightSummary(self.routesByDay, [])
        self.assertEqual(summary["overnight_pairs"],       0)
        self.assertEqual(summary["drivers_in_sleeper"],    0)
        self.assertEqual(summary["per_diem_annual"],       0.0)
        self.assertEqual(summary["sleeper_annual"],        0.0)
        self.assertEqual(summary["overnight_cost_annual"], 0.0)

    def test_milesSavedConsistentWithAnnual(self):
        self.assertEqual(
            self.summary["annual_miles_saved"],
            self.summary["miles_saved_weekly"] * self.cm.weeks_per_year
        )

    def test_annualMileageSavingFormula(self):
        expected = round(
            self.summary["annual_miles_saved"] * self.cm.cost_per_mile_van, 2
        )
        self.assertAlmostEqual(self.summary["annual_mileage_saving"], expected, places=2)

if __name__ == "__main__":
    unittest.main()
