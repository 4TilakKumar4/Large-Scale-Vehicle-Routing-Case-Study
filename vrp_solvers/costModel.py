"""
vrp_solvers/costModel.py — Cost estimation for NHG/MAD weekly routing solutions.

All default rates are sourced from 2024-25 industry benchmarks for Northeast US
dedicated contract carriers:

  Mileage costs    : ATRI 2025 Analysis of Operational Costs (2024 data)
                     fuel $0.481 + R&M $0.198 + tires $0.046 = $0.725/mi (Van)
                     ST approx 13% higher due to heavier vehicle / lift-gate
  Driver wages     : BLS 2024 + Massachusetts ZipRecruiter / Glassdoor data
                     $32/hr baseline for experienced Northeast dedicated CDL driver
  Benefits loading : ATRI benefits/wages ratio 24.7% + NE cost-of-living premium → 30%
  Per diem         : IRS Notice 2024-68, $80/full day (effective Oct 1 2024,
                     stable through Sep 2026)
  Equipment daily  : ATRI truck/trailer payments $0.39/mi + MA rental market
                     (Apple Truck & Trailer, Ryder MA) → $185/day day-cab
  Insurance daily  : Northeast interstate fleet $15K-20K/year/truck (Progressive
                     Commercial, COGO Insurance 2024); $56/day at $15K/268 operating days

Usage:
    from vrp_solvers.costModel import CostModel
    cm      = CostModel()
    weekly  = cm.weeklyCost(routesByDay)
    annual  = cm.annualCost(weekly["total"])
    report  = cm.weeklyBreakdown(routesByDay)
"""

from vrp_solvers.base import DAYS, evaluateRoute
from vrp_solvers.base import evaluateMixedRoute

# Sensitivity parameter ranges (low / default / high) for documentation

SENSITIVITY_RANGES = {
    "cost_per_mile_van":      (0.65,  0.725, 0.85),
    "cost_per_mile_st":       (0.72,  0.820, 0.95),
    "driver_hourly_wage":     (28.00, 32.00, 38.00),
    "overnight_allowance":    (69.00, 80.00, 100.00),
    "day_cab_daily":          (150.0, 185.0, 230.0),
    "insurance_per_day":      (45.00, 56.00, 75.00),
}

class CostModel:
    """
    Parameterised cost model for NHG/MAD weekly routing solutions.
    All monetary values are in USD.

    Cost components per route:
      1. Mileage   — fuel, repair & maintenance, tires (per mile)
      2. Labour    — driver wages × on-duty hours; overtime above regular_shift_hours
      3. Benefits  — benefits_loading fraction applied to gross wages
      4. Equipment — day_cab_daily (+ sleeper_premium_daily if overnight)
      5. Insurance — insurance_per_day per vehicle dispatched
      6. Per diem  — overnight_allowance per overnight pairing

    Call routeCost() for a single route, weeklyCost() for a full solution,
    annualCost() to annualise, and weeklyBreakdown() for a component-level report.
    """

    def __init__(
        self,
        # --- Mileage variable costs (ATRI 2024) ---
        cost_per_mile_van        = 0.725,   # $/mile: fuel + R&M + tires
        cost_per_mile_st         = 0.820,   # $/mile: ST heavier / lift-gate

        # --- Driver wages (BLS + MA market 2024-25) ---
        driver_hourly_wage       = 32.00,   # $/hour base
        overtime_multiplier      = 1.50,    # × base above regular_shift_hours
        regular_shift_hours      = 8.0,     # hours before overtime kicks in

        # --- Benefits (ATRI 2024 ratio + NE premium) ---
        benefits_loading         = 0.30,    # fraction of gross wages

        # --- Overnight / per diem (IRS Notice 2024-68) ---
        overnight_allowance      = 80.00,   # $/overnight: M&IE per diem (Oct 2024)
        sleeper_premium_daily    = 60.00,   # $/day: sleeper cab premium over day-cab

        # --- Equipment daily (ATRI 2024 + MA market) ---
        day_cab_daily            = 185.00,  # $/day: tractor + trailer payment + amortised maint
        st_trailer_premium_daily = 25.00,   # $/day: lift-gate ST trailer premium over dry van

        # --- Insurance (NE interstate fleet, 2024) ---
        insurance_per_day        = 56.00,   # $/day per vehicle: $15K/yr ÷ 268 operating days

        # --- Annualisation ---
        weeks_per_year           = 52,
    ):
        self.cost_per_mile_van        = cost_per_mile_van
        self.cost_per_mile_st         = cost_per_mile_st
        self.driver_hourly_wage       = driver_hourly_wage
        self.overtime_multiplier      = overtime_multiplier
        self.regular_shift_hours      = regular_shift_hours
        self.benefits_loading         = benefits_loading
        self.overnight_allowance      = overnight_allowance
        self.sleeper_premium_daily    = sleeper_premium_daily
        self.day_cab_daily            = day_cab_daily
        self.st_trailer_premium_daily = st_trailer_premium_daily
        self.insurance_per_day        = insurance_per_day
        self.weeks_per_year           = weeks_per_year

    # Core per-route calculations

    def _mileageCost(self, miles, vehicleType="van"):
        rate = self.cost_per_mile_st if vehicleType == "st" else self.cost_per_mile_van
        return round(miles * rate, 2)

    def _labourCost(self, dutyHours):
        """Gross wages for one driver-shift. Overtime above regular_shift_hours."""
        regularHours  = min(dutyHours, self.regular_shift_hours)
        overtimeHours = max(0.0, dutyHours - self.regular_shift_hours)
        grossWages = (regularHours  * self.driver_hourly_wage
                      + overtimeHours * self.driver_hourly_wage * self.overtime_multiplier)
        return round(grossWages, 2)

    def _benefitsCost(self, grossWages):
        return round(grossWages * self.benefits_loading, 2)

    def _equipmentCost(self, vehicleType="van", isSleeper=False):
        """Daily equipment cost for one dispatched vehicle."""
        cost = self.day_cab_daily
        if vehicleType == "st":
            cost += self.st_trailer_premium_daily
        if isSleeper:
            cost += self.sleeper_premium_daily
        return round(cost, 2)

    def routeCost(self, routeResult, vehicleType="van", isSleeper=False):
        """
        Full cost for one route given its evaluated result dict.
        routeResult must contain total_miles and total_duty (both from evaluateRoute /
        evaluateMixedRoute). Returns a dict of cost components and total.
        """
        miles      = routeResult["total_miles"]
        dutyHours  = routeResult["total_duty"]

        mileage    = self._mileageCost(miles, vehicleType)
        labour     = self._labourCost(dutyHours)
        benefits   = self._benefitsCost(labour)
        equipment  = self._equipmentCost(vehicleType, isSleeper)
        insurance  = round(self.insurance_per_day, 2)

        total = round(mileage + labour + benefits + equipment + insurance, 2)

        return {
            "mileage":   mileage,
            "labour":    labour,
            "benefits":  benefits,
            "equipment": equipment,
            "insurance": insurance,
            "per_diem":  0.0,
            "total":     total,
        }

    def overnightRouteCost(self, overnightResult):
        """
        Cost for one overnight pairing (two-day driver shift with a 10-hour break).
        Equipment is one sleeper-cab vehicle for two days. Per diem is one overnight.
        """
        miles     = overnightResult["total_miles"]
        day1Duty  = overnightResult["day1_duty"]
        day2Duty  = overnightResult["day2_duty"]

        mileage   = self._mileageCost(miles, "van")
        labour    = self._labourCost(day1Duty) + self._labourCost(day2Duty)
        benefits  = self._benefitsCost(labour)
        # Two-day route: sleeper-cab for two days
        equipment = self._equipmentCost("van", isSleeper=True) * 2
        insurance = round(self.insurance_per_day * 2, 2)
        per_diem  = round(self.overnight_allowance, 2)

        total = round(mileage + labour + benefits + equipment + insurance + per_diem, 2)

        return {
            "mileage":   mileage,
            "labour":    labour,
            "benefits":  benefits,
            "equipment": equipment,
            "insurance": insurance,
            "per_diem":  per_diem,
            "total":     total,
        }

    # Weekly aggregation

    def weeklyCost(self, routesByDay, overnightPairings=None,
                   vanByDay=None, stByDay=None):
        """
        Total weekly cost across all days and vehicle types.

        routesByDay    : {day → [route, ...]}  (base case / overnight day-cab routes)
        overnightPairings : list of overnight pairing dicts (optional)
        vanByDay / stByDay : {day → [route, ...]} for mixed fleet (optional)

        Returns a flat dict of weekly cost components and total.
        """
        overnightPairings = overnightPairings or []

        # Identify routes consumed by overnight pairings
        usedRoutes = {}
        for pairing in overnightPairings:
            d1, r1 = pairing["day1"], pairing["route1_idx"]
            d2, r2 = pairing["day2"], pairing["route2_idx"]
            usedRoutes.setdefault(d1, set()).add(r1)
            usedRoutes.setdefault(d2, set()).add(r2)

        totals = {"mileage": 0.0, "labour": 0.0, "benefits": 0.0,
                  "equipment": 0.0, "insurance": 0.0, "per_diem": 0.0}

        # Mixed fleet (van + ST separate)
        if vanByDay is not None and stByDay is not None:
            for day in DAYS:
                for route in vanByDay.get(day, []):
                    r = evaluateMixedRoute(route, "van")
                    for k, v in self.routeCost(r, "van").items():
                        if k != "total":
                            totals[k] += v
                for route in stByDay.get(day, []):
                    r = evaluateMixedRoute(route, "st")
                    for k, v in self.routeCost(r, "st").items():
                        if k != "total":
                            totals[k] += v

        else:
            # Base case / overnight day-cab routes
            for day in DAYS:
                consumed = usedRoutes.get(day, set())
                for idx, route in enumerate(routesByDay.get(day, [])):
                    if idx in consumed:
                        continue
                    r = evaluateRoute(route)
                    for k, v in self.routeCost(r, "van").items():
                        if k != "total":
                            totals[k] += v

            # Overnight pairings
            for pairing in overnightPairings:
                for k, v in self.overnightRouteCost(pairing["results"]).items():
                    if k != "total":
                        totals[k] += v

        totals["total"] = round(sum(v for k, v in totals.items() if k != "total"), 2)
        return {k: round(v, 2) for k, v in totals.items()}

    def annualCost(self, weeklyTotal):
        """Annualise a weekly cost figure."""
        return round(weeklyTotal * self.weeks_per_year, 2)

    # Detailed breakdown for reporting

    def weeklyBreakdown(self, routesByDay, overnightPairings=None,
                        vanByDay=None, stByDay=None):
        """
        Return a list of per-route cost dicts plus weekly and annual summaries.
        Each route dict includes day, route_number, vehicle_type, miles, duty_hours,
        and each cost component.
        """
        overnightPairings = overnightPairings or []

        usedRoutes = {}
        for pairing in overnightPairings:
            usedRoutes.setdefault(pairing["day1"], set()).add(pairing["route1_idx"])
            usedRoutes.setdefault(pairing["day2"], set()).add(pairing["route2_idx"])

        rows = []

        if vanByDay is not None and stByDay is not None:
            for day in DAYS:
                for rNum, route in enumerate(vanByDay.get(day, []), start=1):
                    r = evaluateMixedRoute(route, "van")
                    c = self.routeCost(r, "van")
                    rows.append({
                        "day": day, "route_number": rNum,
                        "vehicle_type": "Van", "route_type": "day_cab",
                        "miles": r["total_miles"], "duty_hours": r["total_duty"],
                        **c
                    })
                for rNum, route in enumerate(stByDay.get(day, []), start=1):
                    r = evaluateMixedRoute(route, "st")
                    c = self.routeCost(r, "st")
                    rows.append({
                        "day": day, "route_number": rNum,
                        "vehicle_type": "StraightTruck", "route_type": "day_cab",
                        "miles": r["total_miles"], "duty_hours": r["total_duty"],
                        **c
                    })

        else:
            for day in DAYS:
                consumed = usedRoutes.get(day, set())
                for idx, route in enumerate(routesByDay.get(day, [])):
                    if idx in consumed:
                        continue
                    r = evaluateRoute(route)
                    c = self.routeCost(r, "van")
                    rows.append({
                        "day": day, "route_number": idx + 1,
                        "vehicle_type": "Van", "route_type": "day_cab",
                        "miles": r["total_miles"], "duty_hours": r["total_duty"],
                        **c
                    })

            for k, pairing in enumerate(overnightPairings, start=1):
                res = pairing["results"]
                c   = self.overnightRouteCost(res)
                rows.append({
                    "day": f"{pairing['day1']}-{pairing['day2']}",
                    "route_number": k,
                    "vehicle_type": "Van",
                    "route_type": "overnight",
                    "miles": res["total_miles"],
                    "duty_hours": round(res["day1_duty"] + res["day2_duty"], 3),
                    **c
                })

        weeklyTotals = {"mileage": 0.0, "labour": 0.0, "benefits": 0.0,
                        "equipment": 0.0, "insurance": 0.0, "per_diem": 0.0}
        for row in rows:
            for k in weeklyTotals:
                weeklyTotals[k] += row[k]

        weeklyTotals["total"] = round(sum(weeklyTotals.values()), 2)

        return {
            "routes":  rows,
            "weekly":  {k: round(v, 2) for k, v in weeklyTotals.items()},
            "annual":  {k: round(v * self.weeks_per_year, 2)
                        for k, v in weeklyTotals.items()},
        }

    def overnightSummary(self, baseRoutesByDay, overnightPairings):
        """
        Compare overnight-allowed vs no-overnight costs and miles.

        Returns a dict with:
          miles_saved         : total miles reduction from overnight pairings
          annual_miles_saved  : miles_saved × 52
          overnight_cost      : annual per diem + sleeper cab premium for all pairings
          annual_mileage_saving : annual_miles_saved × cost_per_mile_van
          net_annual_saving   : annual_mileage_saving - overnight_cost
          overnight_pairs     : number of overnight pairings
          drivers_in_sleeper  : same as overnight_pairs (one driver per pairing)
          per_diem_annual     : overnight_allowance × pairs × 52
          sleeper_annual      : sleeper_premium_daily × 2 × pairs × 52

        Positive net_annual_saving means overnight routing is cheaper overall.
        """
        from vrp_solvers.base import evaluateRoute

        usedRoutes = {}
        for p in overnightPairings:
            usedRoutes.setdefault(p["day1"], set()).add(p["route1_idx"])
            usedRoutes.setdefault(p["day2"], set()).add(p["route2_idx"])

        # Miles without overnight (base routes only, no pairings)
        baseMiles = sum(
            evaluateRoute(r)["total_miles"]
            for day in DAYS
            for r in baseRoutesByDay.get(day, [])
        )

        # Miles with overnight (day-cab routes + overnight pairings)
        overnightMiles = (
            sum(
                evaluateRoute(r)["total_miles"]
                for day in DAYS
                for idx, r in enumerate(baseRoutesByDay.get(day, []))
                if idx not in usedRoutes.get(day, set())
            )
            + sum(p["results"]["total_miles"] for p in overnightPairings)
        )

        pairCount          = len(overnightPairings)
        milesSaved         = round(baseMiles - overnightMiles, 0)
        annualMilesSaved   = round(milesSaved * self.weeks_per_year, 0)
        annualMileageSaving = round(annualMilesSaved * self.cost_per_mile_van, 2)

        # Annual cost of overnight operations
        perDiemAnnual  = round(self.overnight_allowance * pairCount * self.weeks_per_year, 2)
        sleeperAnnual  = round(self.sleeper_premium_daily * 2 * pairCount * self.weeks_per_year, 2)
        overnightCost  = round(perDiemAnnual + sleeperAnnual, 2)

        netAnnualSaving = round(annualMileageSaving - overnightCost, 2)

        return {
            "overnight_pairs":        pairCount,
            "drivers_in_sleeper":     pairCount,
            "miles_saved_weekly":     int(milesSaved),
            "annual_miles_saved":     int(annualMilesSaved),
            "annual_mileage_saving":  annualMileageSaving,
            "per_diem_annual":        perDiemAnnual,
            "sleeper_annual":         sleeperAnnual,
            "overnight_cost_annual":  overnightCost,
            "net_annual_saving":      netAnnualSaving,
        }

    def printOvernightSummary(self, overnightSummaryResult):
        """Print a formatted overnight cost-vs-savings framing."""
        s = overnightSummaryResult
        print("\nOvernight Routes — Cost vs Savings Analysis")
        print("-" * 60)
        print(f"  Overnight pairings:              {s['overnight_pairs']}")
        print(f"  Drivers in sleeper cab (weekly): {s['drivers_in_sleeper']}")
        print(f"  Weekly miles saved:              {s['miles_saved_weekly']:,}")
        print(f"  Annual miles saved:              {s['annual_miles_saved']:,}")
        print()
        print(f"  Annual mileage saving:         ${s['annual_mileage_saving']:>10,.2f}")
        print(f"  Annual per diem cost:          ${s['per_diem_annual']:>10,.2f}")
        print(f"  Annual sleeper cab premium:    ${s['sleeper_annual']:>10,.2f}")
        print(f"  Total overnight cost (annual): ${s['overnight_cost_annual']:>10,.2f}")
        print()
        verdict = "NET SAVING" if s['net_annual_saving'] >= 0 else "NET COST"
        print(f"  {verdict}:                     ${abs(s['net_annual_saving']):>10,.2f}/year")
        print()

    def printSummary(self, weeklyBreakdownResult, label="Solution"):
        """Print a formatted cost summary from weeklyBreakdown() output."""
        w = weeklyBreakdownResult["weekly"]
        a = weeklyBreakdownResult["annual"]

        print(f"\nCost Summary — {label}")
        print("-" * 60)
        print(f"  {'Component':<22} {'Weekly':>12}  {'Annual':>12}")
        print(f"  {'-'*22} {'-'*12}  {'-'*12}")

        labels = {
            "mileage":   "Mileage (fuel+R&M)",
            "labour":    "Driver wages",
            "benefits":  "Driver benefits",
            "equipment": "Equipment",
            "insurance": "Insurance",
            "per_diem":  "Per diem (overnight)",
        }
        for key, lbl in labels.items():
            print(f"  {lbl:<22} ${w[key]:>11,.2f}  ${a[key]:>11,.2f}")

        print(f"  {'-'*22} {'-'*12}  {'-'*12}")
        print(f"  {'TOTAL':<22} ${w['total']:>11,.2f}  ${a['total']:>11,.2f}")
        print()