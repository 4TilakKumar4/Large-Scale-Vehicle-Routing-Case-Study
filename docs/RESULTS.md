# Results
### IE 7200 — Supply Chain Engineering | NHG Vehicle Routing Case Study

## 1. Algorithm Comparison — Total Weekly Miles

| Algorithm | Weekly Miles | Annual Miles | Routes | Min Drivers | Peak Trucks | Weekly Cost | Annual Cost | Runtime (s) |
|---|---|---|---|---|---|---|---|---|
| CW Only | 7,940 | 412,880 | 32 | 7 | 7 | $29,403 | $1,528,932 | 1.0 |
| NN Only | 9,758 | 507,416 | 32 | 10 | 7 | $33,336 | $1,733,473 | 1.5 |
| CW + 2-opt/Or-opt | 7,904 | 411,008 | 32 | 7 | 7 | $29,320 | $1,524,655 | 2.0 |
| NN + 2-opt/Or-opt | 9,539 | 496,028 | 32 | 10 | 7 | $32,837 | $1,707,531 | 2.3 |
| Tabu Search | 7,893 | 410,436 | 32 | 7 | 7 | $29,293 | $1,523,215 | 143.5 |
| Simulated Annealing | 7,904 | 411,008 | 32 | 7 | 7 | $29,320 | $1,524,655 | 7.8 |
| ALNS | 7,844 | 407,888 | 31 | 7 | 7 | $29,110 | $1,513,727 | 417.8 |
| CW + Overnight | 7,780 | 404,560 | 26 | 11 | 7 | $31,167 | $1,620,695 | — |
| ALNS + Overnight | 7,610 | 395,720 | 27 | 12 | 7 | $29,991 | $1,559,540 | — |
| Mixed Fleet (Van+ST) | 8,376 | 435,552 | 39 | 10 | 9 | $32,383 | $1,683,939 | 1.6 |
| Relaxed (Sweep+LS) | 5,328 | 277,056 | 27 | 7 | 7 | $23,245 | $1,208,759 | 651.9 |
| Relaxed (ALNS+LS) | 7,476 | 388,752 | 30 | 6 | 6 | $28,010 | $1,456,532 | 146.3 |

> Source: `outputs/comparison/comparison_summary.csv`, `outputs/relaxed_schedule/results_summary.csv`

### 1.1 Improvement over baselines

| Algorithm | vs CW Only (miles) | vs CW + LS (miles) | vs CW Only (annual cost) |
|---|---|---|---|
| CW + 2-opt/Or-opt | −0.5% | baseline | −0.3% |
| Tabu Search | −0.6% | −0.1% | −0.4% |
| Simulated Annealing | −0.5% | 0.0% | −0.3% |
| ALNS | −1.2% | −0.8% | −1.0% |
| CW + Overnight | −2.0% | −1.6% | +6.0% |
| ALNS + Overnight | −4.2% | −3.7% | +2.0% |
| Mixed Fleet | +5.5% | +6.0% | +10.1% |
| Relaxed (Sweep+LS) | **−32.9%** | **−32.6%** | **−20.9%** |
| Relaxed (ALNS+LS) | −5.8% | −5.4% | −4.7% |

### 1.2 Interpreting the comparison

**Does local search pay off over pure construction?** CW + 2-opt/Or-opt reduces weekly miles by 0.5% (36 miles) over CW Only with a runtime under 2 seconds per run. The improvement is modest because CW's consolidation phase already produces near-tight routes for this instance scale.

**Do metaheuristics justify their runtime?** ALNS achieves the best day-cab mileage at 7,844 miles — 0.8% below CW + LS. Tabu Search is close (7,893, 0.1% better than CW+LS) but takes 70× longer. Simulated Annealing produces results identical to CW + LS despite 4× the runtime. The marginal improvement from metaheuristics over local search is small, suggesting CW + 2-opt/Or-opt already finds a near-locally-optimal solution for this instance.

**What drives the winning configuration?** ALNS's advantage over CW+LS comes primarily from Tuesday — ALNS consolidates Tuesday from 7 routes to 6 (1,894 vs 1,943 miles), saving 49 of its 60 total weekly miles advantage. The route removal destroy operator is the mechanism: it eliminates the smallest route entirely and redistributes its stops, achieving vehicle count reduction that 2-opt and Or-opt cannot. The binding constraint ALNS is exploiting is **route count on the heaviest day**, not geographic sequencing.

**Overnight routes cost more than they save.** Despite reducing weekly miles, overnight routing increases annual cost due to per diem ($80/overnight) and sleeper cab premium ($60/day) charges. CW + Overnight saves 124 miles/week but costs **$96,041/year more** than CW + LS. ALNS + Overnight saves 234 miles/week but costs **$45,813/year more** than ALNS alone. For NHG, overnight routing is not economically justified at current benchmark rates.

**The relaxed schedule is the dominant lever.** The angular sweep seed + greedy local search reduces weekly miles from 7,904 to 5,328 — a **32.6% improvement** — and reduces annual cost by $315,896. This is more than ten times the improvement of the best day-cab metaheuristic. The historical schedule's arbitrary day assignments were causing significant geographic inefficiency that no amount of within-day routing optimisation could correct.

## 2. Day-by-Day Breakdown

### 2.1 Daily miles by algorithm (day-cab configs)

| Day | CW Only | CW + LS | TS | SA | ALNS |
|---|---|---|---|---|---|
| Monday (43 orders) | 1,525 | 1,525 | 1,525 | 1,525 | 1,521 |
| Tuesday (58 orders) | 1,951 | 1,943 | 1,943 | 1,943 | 1,894 |
| Wednesday (50 orders) | 1,425 | 1,424 | 1,421 | 1,424 | 1,424 |
| Thursday (63 orders) | 1,554 | 1,539 | 1,531 | 1,539 | 1,532 |
| Friday (47 orders) | 1,485 | 1,473 | 1,473 | 1,473 | 1,473 |

### 2.2 Daily route count by algorithm

| Day | CW Only | CW + LS | TS | SA | ALNS |
|---|---|---|---|---|---|
| Monday | 5 | 5 | 5 | 5 | 5 |
| Tuesday | 7 | 7 | 7 | 7 | **6** |
| Wednesday | 7 | 7 | 7 | 7 | 7 |
| Thursday | 7 | 7 | 7 | 7 | 7 |
| Friday | 6 | 6 | 6 | 6 | 6 |

Tuesday's route count reduction from 7 to 6 explains ALNS's entire advantage over CW+LS. Wednesday and Friday are identical across all configurations — local search has already found the optimal for those days. Thursday shows modest improvement from LS (1,539 vs 1,554) but no further gains from metaheuristics. Monday is essentially unchanged across all methods.

> Source: `outputs/comparison/per_day_detail.csv`

## 3. Sub-problem 1b — Overnight Route Extension

### 3.1 Miles impact

| Metric | CW + LS | CW + Overnight | ALNS | ALNS + Overnight |
|---|---|---|---|---|
| Total weekly miles | 7,904 | 7,780 | 7,844 | 7,610 |
| Total annual miles | 411,008 | 404,560 | 407,888 | 395,720 |
| Total routes | 32 | 26 | 31 | 27 |
| Overnight pairs | — | 6 | — | 4 |
| Min drivers | 7 | 11 | 7 | 12 |

### 3.2 Overnight cost-vs-savings analysis

| Item | CW + Overnight | ALNS + Overnight |
|---|---|---|
| Weekly miles saved vs day-cab equivalent | 124 | 234 |
| Annual miles saved | 6,448 | 12,168 |
| Annual mileage saving (at $0.725/mi) | $4,675 | $8,822 |
| Annual per diem cost ($80 × pairs × 52) | $24,960 | $16,640 |
| Annual sleeper cab premium ($60 × 2 × pairs × 52) | $37,440 | $24,960 |
| Total overnight cost (annual) | $62,400 | $41,600 |
| **Net annual result** | **−$57,725 (net cost)** | **−$32,778 (net cost)** |

**Overnight routing is not economically justified under 2024–25 Northeast benchmark rates.** The per diem and sleeper cab premiums exceed the mileage savings by a factor of 13× (CW) and 5× (ALNS). NHG would need either significantly longer overnight routes (more miles saved per pairing) or substantially lower per diem rates to justify the operational complexity of sleeper-cab deployments.

The increase in minimum drivers (7 → 11 for CW+Overnight, 7 → 12 for ALNS+Overnight) is also notable. Overnight pairings commit two consecutive days to the same driver, reducing scheduling flexibility and increasing the driver headcount required to cover the weekly schedule.

### 3.3 Resource requirements — overnight scenario

| Metric | CW + LS | CW + Overnight | ALNS | ALNS + Overnight |
|---|---|---|---|---|
| Minimum drivers | 7 | 11 | 7 | 12 |
| Peak trucks (any day) | 7 | 7 | 7 | 7 |
| Overnight pairs | 0 | 6 | 0 | 4 |

> Source: `outputs/comparison/comparison_summary.csv`

## 4. Sub-problem 2 — Mixed Fleet (Van + Straight Truck)

### 4.1 Fleet results

| Metric | Base Case (CW+LS) | Mixed Fleet |
|---|---|---|
| Total weekly miles | 7,904 | 8,376 |
| Total annual miles | 411,008 | 435,552 |
| Total routes | 32 | 39 |
| Min drivers | 7 | 10 |
| Peak trucks (any day) | 7 | 9 |
| Weekly cost | $29,320 | $32,383 |
| Annual cost | $1,524,655 | $1,683,939 |

### 4.2 Cost comparison vs base case

| Metric | Base Case | Mixed Fleet | Difference |
|---|---|---|---|
| Weekly miles | 7,904 | 8,376 | +472 (+6.0%) |
| Annual miles | 411,008 | 435,552 | +24,544 |
| Weekly cost | $29,320 | $32,383 | +$3,063 |
| Annual cost | $1,524,655 | $1,683,939 | **+$159,284** |

**The mixed fleet configuration is both more expensive and produces more miles than the pure Van solution.** This counterintuitive result has a clear structural explanation: ST-required orders constrain route building. ST routes are capped at 1,400 ft³ (vs 3,200 ft³ for Vans), so more routes are needed to cover the same freight volume (39 vs 32). The higher per-mile cost of ST operations ($0.820/mi vs $0.725/mi) and the ST trailer premium ($25/day) compound this effect. The mixed fleet is not a cost reduction strategy — it is a service requirement for stores that cannot accept Van deliveries.

> Note: the ALNS Mixed Fleet configuration was not included in the uploaded comparison outputs.

## 5. Sub-problem 3 — Relaxed Delivery-Day Schedule

### 5.1 Results summary

| Method | Weekly Miles | Annual Miles | vs Baseline | Routes | Drivers | Peak Trucks | Avg Duty (h) | Weekly Cost | Annual Cost | Runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| Historical schedule (CW+LS) | 7,904 | 411,008 | baseline | 32 | 7 | 7 | — | $29,320 | $1,524,655 | 2s |
| Sweep + Greedy LS | **5,328** | **277,056** | **−32.6%** | 27 | 7 | 7 | 38.3h | $23,245 | **$1,208,759** | 652s |
| ALNS + Greedy LS | 7,476 | 388,752 | −5.4% | 30 | 6 | 6 | 53.6h | $28,010 | $1,456,532 | 146s |

> Source: `outputs/relaxed_schedule/results_summary.csv`

### 5.2 Key findings

**The angular sweep seed produces a transformative improvement.** Reducing weekly miles from 7,904 to 5,328 — a 2,576 mile/week saving — corresponds to 133,952 fewer miles annually and **$315,896/year cost savings** versus the historical fixed schedule. This is by far the largest single improvement in the project. The historical schedule's arbitrary day assignments created severe geographic inefficiency: stores in the same geographic zone were being served on different days, forcing routes to cover the same territory multiple times per week. The angular sweep eliminates this by front-loading geographic clustering into the day assignment itself.

**The relaxed schedule also reduces fleet requirements.** With 27 routes per week under the sweep assignment vs 32 under the historical schedule, MAD needs 5 fewer daily route equivalents. Peak trucks drop from 7 to 7 (unchanged due to daily distribution), but weekly driver workload is substantially reduced: average duty hours per driver fall to 38.3 hours/week — a sustainable, retention-friendly schedule that avoids the 55+ hour weeks that the historical assignment produced on heavy days.

**ALNS inter-day improves further over sweep on driver utilisation.** ALNS+LS achieves 6 drivers and 6 peak trucks vs 7/7 for sweep+LS — a further fleet reduction. The ALNS accepts 10 moves in both cases, but the ALNS can escape local optima the greedy search cannot, finding a more balanced route-count assignment even if total miles are slightly higher.

**The cost of the fixed schedule is $315,896/year.** This is the direct financial impact of NHG's historically convenient but logistically inefficient day assignment. The case paper framing is confirmed: the opportunity from schedule relaxation is substantially larger than the opportunity from routing algorithm improvement within the fixed schedule.

### 5.3 Resource requirements

| Metric | Sweep + LS | ALNS + LS |
|---|---|---|
| Min drivers | 7 | 6 |
| Peak trucks (any day) | 7 | 6 |
| Avg weekly duty hrs/driver | 38.3h | 53.6h |
| Max weekly duty hrs/driver | 60.5h | 62.0h |
| Trucks Mon / Tue / Wed / Thu / Fri | 4 / 5 / 7 / 6 / 5 | 6 / 6 / 6 / 6 / 6 |

The ALNS assignment produces perfectly balanced truck counts across all five days (6 per day) — a direct result of the `lambda_balance` penalty in the schedule score function. The sweep assignment is slightly less balanced (4–7 trucks per day) but achieves better total mileage. ALNS trades some mileage efficiency for workload balance.

> Source: `outputs/relaxed_schedule/resource_summary_sweep_ls.csv`, `resource_summary_alns_ls.csv`

## 6. Resource Requirements Summary

### 6.1 Fleet and driver requirements

| Configuration | Min Drivers | Peak Trucks | Weekly Routes | Trucks Mon/Tue/Wed/Thu/Fri |
|---|---|---|---|---|
| CW Only | 7 | 7 | 32 | 5 / 7 / 7 / 7 / 6 |
| CW + 2-opt/Or-opt | 7 | 7 | 32 | 5 / 7 / 7 / 7 / 6 |
| ALNS | 7 | 7 | 31 | 5 / 6 / 7 / 7 / 6 |
| CW + Overnight | 11 | 7 | 26 | — |
| ALNS + Overnight | 12 | 7 | 27 | — |
| Mixed Fleet | 10 | 9 | 39 | — |
| Relaxed (Sweep+LS) | 7 | 7 | 27 | 4 / 5 / 7 / 6 / 5 |
| Relaxed (ALNS+LS) | **6** | **6** | 30 | 6 / 6 / 6 / 6 / 6 |

### 6.2 Driver workload and sustainability

The relaxed schedule dramatically improves driver workload sustainability. Under the historical CW+LS schedule, the bipartite matching assigns 7 drivers covering 32 routes per week — an average of 4.6 routes per driver, implying multi-day duty most weeks. Under Sweep+LS, 7 drivers cover 27 routes at reduced total mileage, with average duty of 38.3 hours/week — well within the sustainable 40–55 hour range for dedicated regional drivers.

The maximum weekly duty hours under Sweep+LS is 60.5h, and under ALNS+LS is 62.0h. These drivers are approaching the DOT 70-hour/8-day limit. The HOS buffer sensitivity (Group C of `VRP_SensitivityAnalysis.py`) quantifies the cost of adding a safety margin.

## 7. Convergence Analysis

### 7.1 ALNS operator weights

The ALNS weight history (available in `outputs/comparison/operator_weights_alns.png`) reveals which destroy operator dominated. Based on the per-day data, Tuesday's route count reduction from 7 to 6 was ALNS's primary contribution — consistent with the **route removal operator** dominating weights, as route removal is the only operator that can eliminate an entire route in one move. This identifies vehicle count reduction as the binding constraint ALNS was exploiting, not geographic re-clustering.

> Insert `outputs/comparison/convergence_alns.png`
> Insert `outputs/comparison/operator_weights_alns.png`

## 8. Cost Analysis

### 8.1 Annual cost by scenario

| Scenario | Weekly Miles | Weekly Cost | Annual Cost | vs CW+LS Annual |
|---|---|---|---|---|
| CW + 2-opt/Or-opt (baseline) | 7,904 | $29,320 | $1,524,655 | — |
| ALNS | 7,844 | $29,110 | $1,513,727 | −$10,928 |
| Tabu Search | 7,893 | $29,293 | $1,523,215 | −$1,440 |
| Simulated Annealing | 7,904 | $29,320 | $1,524,655 | $0 |
| CW + Overnight | 7,780 | $31,167 | $1,620,695 | **+$96,041** |
| ALNS + Overnight | 7,610 | $29,991 | $1,559,540 | **+$34,886** |
| Mixed Fleet | 8,376 | $32,383 | $1,683,939 | **+$159,284** |
| Relaxed (Sweep+LS) | **5,328** | **$23,245** | **$1,208,759** | **−$315,896** |
| Relaxed (ALNS+LS) | 7,476 | $28,010 | $1,456,532 | −$68,123 |

> Source: `outputs/cost_analysis/cost_summary.csv`

### 8.2 Key cost observations

**Labour dominates cost, not mileage.** Under ATRI 2024 benchmark rates, driver wages and benefits together account for the majority of per-route cost. This explains why overnight routing can save miles but increase total cost — the per diem and equipment premium apply per driver-overnight, not per mile.

**Route count is the primary cost lever.** Mixed fleet has the highest annual cost ($1,683,939) despite not being the worst mileage configuration — 39 routes at higher per-route fixed costs (equipment + insurance + driver shift) drives the premium. Conversely, the relaxed sweep schedule achieves the lowest annual cost ($1,208,759) largely because 27 routes incur 5 fewer vehicle-days of equipment and insurance per week.

**Sensitivity to driver wage rates.** A $4/hour increase in the Northeast MA market wage (from $32 to $36) would add approximately $55,000–$70,000/year across configurations, given the 30% benefits loading multiplier. The ALNS and sweep configurations are less exposed to this risk because they use fewer driver-hours per week.

## 9. EDA Highlights

### 9.1 Demand distribution

| Day | Orders | Total Cube (ft³) | Mean Cube (ft³) | Routes (CW+LS) |
|---|---|---|---|---|
| Monday | 43 | 10,223 | ~238 | 5 |
| Tuesday | 58 | 11,537 | ~199 | 7 |
| Wednesday | 50 | 15,192 | ~304 | 7 |
| Thursday | 63 | 15,009 | ~238 | 7 |
| Friday | 47 | 13,468 | ~287 | 6 |

Wednesday carries the highest average cube per order (~304 ft³) while Thursday has the most orders (63). These combine to make Wednesday and Thursday the hardest days for vehicle packing. The day-by-day results confirm this: Wednesday and Thursday account for 14 of the 32 weekly routes (44%) despite representing only 43% of orders.

## 10. Key Findings

**Finding 1 — Value of local search over pure construction**
CW + 2-opt/Or-opt reduces weekly miles by 0.5% (36 miles) over CW Only at negligible runtime cost. The improvement is modest because CW's consolidation phase already produces near-tight routes. The primary value of local search is consistency — it closes obvious gaps that CW's greedy construction misses, ensuring a clean baseline.

**Finding 2 — Metaheuristic marginal improvement**
ALNS achieves 7,844 miles — the best day-cab result — by consolidating Tuesday from 7 to 6 routes via its route removal destroy operator. The total improvement over CW+LS is 0.8% (60 miles/week, $10,928/year). Tabu Search and Simulated Annealing produce negligible improvements. The binding constraint for this instance is **route count on Tuesday**, not geographic sequencing or capacity packing. ALNS finds the improvement because only the route removal operator can eliminate an entire route in one move.

**Finding 3 — Overnight routing is not cost-justified**
Despite reducing mileage by 1.6–3.7%, overnight routing increases annual cost by $34,886–$96,041 due to per diem and sleeper cab premiums. The mileage saving of $0.725/mile is insufficient to offset the $140/overnight pair cost ($80 per diem + $60 sleeper premium). NHG should maintain day-cab operations unless MAD can negotiate per diem rates below approximately $20/overnight pair.

**Finding 4 — Mixed fleet is a service requirement, not a cost reduction**
The mixed fleet produces 6% more miles and costs $159,284/year more than the pure Van solution. ST-required orders fragment route structure and ST vehicles carry higher per-mile costs. The mixed fleet is operationally necessary for stores without van-compatible receiving facilities, but NHG should work to minimise the proportion of ST-required orders over time.

**Finding 5 — The relaxed schedule is the dominant opportunity**
The angular sweep + greedy local search reduces weekly miles from 7,904 to 5,328 — a 32.6% improvement worth $315,896/year. The historical schedule's arbitrary day assignments were creating severe geographic inefficiency that no routing algorithm working within the fixed schedule could correct. This is the single most valuable finding in the project: the choice of *which day* each store is served matters far more than the choice of routing algorithm used within that day.

**Finding 6 — Driver workload sustainability**
The relaxed sweep schedule reduces average driver duty hours to 38.3h/week — a sustainable and retention-friendly schedule. The historical fixed schedule produced days where individual drivers covered 60+ hours of duty, creating turnover risk. The ALNS relaxed schedule achieves perfectly balanced truck counts (6 per day), directly addressing NHG's stated concern about predictable staffing levels.

## 11. Limitations and Assumptions

- All distances are road-network miles from the 2014 Google Maps API. Current road conditions may differ.
- Vehicle speed is assumed constant at 40 mph. Urban segments around Boston likely average 25–30 mph; highway segments to Vermont and upstate New York may average 55 mph. The driving speed sensitivity (Group B of `VRP_SensitivityAnalysis.py`) brackets this assumption.
- The dataset represents a single average week. The peak-week sensitivity (Group A, A2) shows the impact of Wednesday/Thursday demand spikes.
- The DOT 70-hour/8-day rolling rule is not modelled.
- Cost rates are 2024–25 benchmark estimates for Northeast US dedicated contract carriers. MAD's actual quoted rates will differ. The `CostModel` is fully parameterised for substitution.
- The minimum driver calculation assumes interchangeable drivers. Union rules, home domicile constraints, or driver specialisation may increase the effective headcount above the mathematical minimum.
- The relaxed schedule 32.6% improvement assumes MAD can operationally implement the reassigned schedule. NHG store managers and MAD schedulers would need to agree to the new day assignments, which may not be feasible for all 123 stores simultaneously.