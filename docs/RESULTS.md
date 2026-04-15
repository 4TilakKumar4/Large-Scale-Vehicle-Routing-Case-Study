# Results
### IE 7200 — Supply Chain Engineering | NHG Vehicle Routing Case Study

> **Status:** Template — tables and findings to be completed after final solver runs.

---

## 1. Weekly Solution Summary

### 1.1 Algorithm Comparison — Total Weekly Miles

| Algorithm | Weekly Miles | Routes | Min Drivers | Peak Trucks | Runtime (s) |
|---|---|---|---|---|---|
| CW Only | | | | | |
| NN Only | | | | | |
| CW + 2-opt/Or-opt | | | | | |
| NN + 2-opt/Or-opt | | | | | |
| Tabu Search | | | | | |
| Simulated Annealing | | | | | |
| ALNS | | | | | |
| CW + Overnight | | | | | |
| ALNS + Overnight | | | | | |

> Source: `outputs/comparison/comparison_summary.csv`

### 1.2 Improvement Over Baseline

| Algorithm | Miles vs CW Only | Miles vs CW + LS | Drivers vs CW + LS |
|---|---|---|---|
| CW + 2-opt/Or-opt | — | baseline | baseline |
| Tabu Search | | | |
| Simulated Annealing | | | |
| ALNS | | | |
| CW + Overnight | | | |
| ALNS + Overnight | | | |

---

## 2. Day-by-Day Breakdown

### 2.1 Daily Miles by Algorithm

| Day | CW Only | CW + LS | TS | SA | ALNS | CW+ON | ALNS+ON |
|---|---|---|---|---|---|---|---|
| Monday | | | | | | | |
| Tuesday | | | | | | | |
| Wednesday | | | | | | | |
| Thursday | | | | | | | |
| Friday | | | | | | | |

> Source: `outputs/comparison/per_day_detail.csv`

### 2.2 Daily Route Count by Algorithm

| Day | CW Only | CW + LS | TS | SA | ALNS | CW+ON | ALNS+ON |
|---|---|---|---|---|---|---|---|
| Monday | | | | | | | |
| Tuesday | | | | | | | |
| Wednesday | | | | | | | |
| Thursday | | | | | | | |
| Friday | | | | | | | |

---

## 3. Resource Requirements

### 3.1 Fleet and Driver Summary

| Algorithm | Min Drivers | Peak Trucks | Overnight Pairs | Sleeper Cabs |
|---|---|---|---|---|
| CW + 2-opt/Or-opt | | | 0 | 0 |
| ALNS | | | 0 | 0 |
| CW + Overnight | | | | |
| ALNS + Overnight | | | | |

### 3.2 Driver Schedule — Best Algorithm

> To be completed. Paste driver chain report from `analyser.printReport()` output.

```
Driver chains (ALNS + Overnight):

  Driver  1:
  Driver  2:
  ...
```

---

## 4. Convergence Analysis

### 4.1 Tabu Search

> Insert `outputs/comparison/convergence_tabu.png`

Key observations:
-
-

### 4.2 Simulated Annealing

> Insert `outputs/comparison/convergence_sa.png`

Key observations:
-
-

### 4.3 ALNS

> Insert `outputs/comparison/convergence_alns.png`

Key observations:
-
-

### 4.4 ALNS Operator Weights

> Insert `outputs/comparison/operator_weights_alns.png`

Key observations:
-
-

---

## 5. EDA Highlights

### 5.1 Demand Distribution

> Insert `outputs/eda/demand_by_day.png`

- Heaviest day: Wednesday / Thursday ( ft³)
- Lightest day: Monday ( ft³)
- Mean order cube: ft³ | Median: ft³

### 5.2 Geographic Distribution

> Insert `outputs/eda/geographic_scatter.png`

- States served:
- Furthest stores from depot:
- Store clusters identified:

---

## 6. Key Findings

> To be completed after final run. Suggested structure:

**Finding 1 — Value of local search over pure construction**

**Finding 2 — Metaheuristic improvement over local search**

**Finding 3 — Overnight routing impact on miles and drivers**

**Finding 4 — Day-of-week routing difficulty**

**Finding 5 — Algorithm runtime vs. solution quality tradeoff**

---

## 7. Limitations and Assumptions

- All distances are road-network miles from 2014 Google Maps API data. Current road conditions may differ.
- Vehicle speed is assumed constant at 40 mph across all route types. Urban sections around Boston likely run slower; highway sections to Vermont and New York faster.
- The dataset represents a single average week. Seasonal demand variation is not modelled.
- The DOT 70-hour/8-day rule is not modelled. The dataset represents one week; inter-week driver continuity is outside scope.
- Overnight routes require sleeper-cab trailers. The incremental equipment cost of sleeper vs. day-cab is not quantified here.
- All stores are assumed to be available for delivery during the full 08:00–18:00 window. Individual store receiving constraints are not modelled.
