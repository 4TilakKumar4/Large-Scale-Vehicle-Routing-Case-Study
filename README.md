# Large-Scale Vehicle Routing: NHG Case Study
### IE 7200 — Supply Chain Engineering | Spring 2026
Authors: Tathya Malav Kamadar, Tilak Kumar Byradenahalli Ramesh, Uriel Baron

---

## Overview

This project implements and compares multiple solution methodologies for a large-scale Capacitated Vehicle Routing Problem (CVRP) with time windows and DOT Hours-of-Service constraints. The problem is drawn from the *Growing Pains* teaching case (Milburn, Kirac, and Hadianniasar, 2017), set in the context of Northeastern Home Goods (NHG) and their proposed carrier Massachusetts Area Distribution (MAD), operating out of a single distribution center in Wilmington, Massachusetts.

The objective is to construct a minimum-cost set of weekly delivery routes serving 123 store locations across six northeastern states, subject to:

- Vehicle capacity constraints (3,200 ft³ per van)
- Store delivery time windows (08:00–18:00)
- DOT Hours-of-Service regulations (11-hour drive limit, 14-hour duty limit, 10-hour mandatory break)
- Fixed weekly delivery schedules (262 total orders across five weekdays)

---

## Repository Structure

```
project/
│
├── deliveries.xlsx                  ← Raw input: order table and location table
├── distances.xlsx                   ← Raw input: 124×124 road distance matrix (miles)
│
├── VRP_DataAnalysis.py              ← Phase 1: data cleaning, EDA, and CSV export
├── VRP_BaseCase.py                  ← Phase 2: base case solver (CW + local search)
├── VRP_BaseCase_Map.py              ← Phase 2 variant: base case with Folium map output
├── VRP_OvernightRoutes.py           ← Phase 3: overnight DOT break scenario (standalone)
├── VRP_SolverComparison.py          ← Phase 4: multi-algorithm comparison
│
├── vrp_solvers/                     ← Custom solver package
│   ├── __init__.py
│   ├── base.py                      ← Shared constants, data I/O, route evaluation,
│   │                                   local search utilities
│   ├── clarkeWright.py              ← ClarkeWrightSolver
│   ├── nearestNeighbor.py           ← NearestNeighborSolver
│   ├── tabuSearch.py                ← TabuSearchSolver
│   ├── simulatedAnnealing.py        ← SimulatedAnnealingSolver
│   └── alns.py                      ← ALNSSolver
│
├── data/                            ← Produced by VRP_DataAnalysis.py (not committed)
│   ├── orders_clean.csv
│   ├── locations_clean.csv
│   └── distance_matrix.csv
│
└── outputs/                         ← All generated plots and CSVs (not committed)
    ├── eda/                         ← EDA plots and summary tables
    ├── comparison/                  ← Algorithm comparison plots and CSVs
    └── routes_map.html              ← Interactive Folium route map
```

---

## Run Order and Dependencies

`VRP_DataAnalysis.py` must be executed first. It reads the raw Excel inputs, performs all cleaning and validation, and writes the processed data to `data/`. Every downstream script reads exclusively from `data/` — none of the solver scripts access the raw Excel files directly.

```
deliveries.xlsx ──┐
distances.xlsx  ──┴──► VRP_DataAnalysis.py ──► data/
                                                  │
                        ┌─────────────────────────┘
                        ▼
                   vrp_solvers/base.py (loadInputs reads from data/)
                        │
           ┌────────────┼─────────────────────┐
           ▼            ▼                     ▼
    VRP_BaseCase   VRP_BaseCase_Map   VRP_SolverComparison
           │            │                     │
           ▼            ▼                     ▼
      (console)   outputs/routes_map   outputs/comparison/

VRP_OvernightRoutes.py ──► reads data/ directly, writes to console only
```

To reproduce all results from a clean state:

```bash
python VRP_DataAnalysis.py        # always run first
python VRP_BaseCase.py            # base case routes and console report
python VRP_BaseCase_Map.py        # base case + interactive HTML map
python VRP_SolverComparison.py    # all algorithm configurations (~10–20 min)
python VRP_OvernightRoutes.py     # overnight scenario, independent of above
```

> **Note:** All scripts must be run from the project root directory. The `vrp_solvers` package is resolved relative to the working directory, so launching scripts from any other location will cause import errors.

---

## Methodology

### Problem Formulation

The problem is modelled as a CVRP with time windows (CVRPTW) and HOS constraints. Each weekday is treated independently: orders assigned to a given day must be served on that day, and routes may not span multiple days except in the overnight extension (Sub-problem 3). The depot is located at ZIP code 01887 (Wilmington, MA). All distances are road-network miles sourced from the Google Maps API (2014).

Route feasibility is evaluated against three simultaneous constraints:

| Constraint | Limit |
|---|---|
| Vehicle capacity | ≤ 3,200 ft³ |
| Delivery window | 08:00–18:00 each stop |
| DOT driving | ≤ 11 hours per shift |
| DOT on-duty | ≤ 14 hours per shift |

Vehicle dispatch time is back-calculated so that the driver arrives at the first stop exactly at 08:00. Unload time per stop is `max(30 min, 0.03 min/ft³ × cube)`. Loading at the DC does not count toward on-duty hours.

### Construction Heuristics

**Clarke-Wright Savings (CW)** — The algorithm initialises one route per order (depot → store → depot) and iteratively merges pairs of routes in descending order of savings value:

$$s(i, j) = d(\text{depot}, i) + d(\text{depot}, j) - d(i, j)$$

Merges are accepted only when all four feasibility constraints remain satisfied. Four merge orientations are attempted at each step to handle both forward and reversed adjacency.

**Nearest Neighbor (NN)** — Routes are extended greedily by appending the closest unvisited stop that keeps the current route feasible. A new route is opened from the depot whenever no feasible extension exists.

Both construction heuristics are followed by a **route consolidation** phase, which attempts to eliminate the smallest route by redistributing its stops via cheapest feasible insertion into the remaining routes. This is repeated until no further reduction is possible.

### Local Search

Two intra-route local search operators are applied after construction as a polishing step:

**2-opt** — All pairs of edges within a route are considered for reversal. A reversal is accepted when it reduces total route distance without violating feasibility. The process repeats until no improving 2-opt move exists.

**Or-opt** — Chains of 1, 2, or 3 consecutive stops are relocated to their cheapest feasible position within the same route. Or-opt is strictly more general than 2-opt for small chains and complements it by finding moves 2-opt cannot reach.

The construction + local search pipeline represents the baseline for comparison and is the solution reported in the base case.

### Metaheuristics

All three metaheuristics are seeded from the Clarke-Wright + consolidation + 2-opt + Or-opt solution, ensuring a fair comparison against a locally optimal starting point.

**Tabu Search (TS)** — At each iteration, the best non-tabu move from a relocate/swap neighbourhood is accepted, even if it worsens the current solution. A tabu list of fixed tenure prevents revisiting recently explored moves. An aspiration criterion overrides tabu status when a global best solution is found.

**Simulated Annealing (SA)** — A single random stop relocation is proposed at each iteration. Improving moves are always accepted; worsening moves are accepted with probability $\exp(-\Delta / T)$, where $T$ is reduced geometrically according to a cooling schedule. This probabilistic acceptance enables escape from local optima.

**Adaptive Large Neighborhood Search (ALNS)** — At each iteration, a destroy operator removes a subset of stops and a repair operator reinserts them. Four destroy operators are maintained (random removal, worst removal, Shaw/related removal, and route removal) alongside two repair operators (greedy insertion and regret-2 insertion). Operator selection uses roulette-wheel weighting; weights are updated periodically based on the quality of improvements each operator has historically produced. A simulated annealing acceptance criterion governs whether new solutions replace the incumbent.

### Algorithm Comparison Matrix

| Configuration | Construction | Local Search | Metaheuristic |
|---|---|---|---|
| `cw_only` | Clarke-Wright | — | — |
| `nn_only` | Nearest Neighbor | — | — |
| `cw_2opt_oropt` | Clarke-Wright | 2-opt + Or-opt | — |
| `nn_2opt_oropt` | Nearest Neighbor | 2-opt + Or-opt | — |
| `tabu_search` | CW | 2-opt + Or-opt (seed) | Tabu Search |
| `simulated_annealing` | CW | 2-opt + Or-opt (seed) | Simulated Annealing |
| `alns` | CW | 2-opt + Or-opt (seed) | ALNS |

### Overnight Extension

`VRP_OvernightRoutes.py` implements the Sub-problem 1 overnight scenario. When a driver exhausts their drive or duty hours before returning to the depot, the DOT-mandated 10-hour break is taken as late as legally possible — the driver continues toward the next day's first stop until the HOS limit is reached, takes the break at that point, and resumes once the break ends and the delivery window opens. Day-1 and day-2 HOS limits are tracked independently. Overnight routes require a sleeper cab and are evaluated separately from the standard day routes.

---

## Dependencies

```
pandas
numpy
matplotlib
folium
pgeocode
scikit-learn
openpyxl
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib folium pgeocode scikit-learn openpyxl
```

---

## Reference

Milburn, A.B., Kirac, E., and Hadianniasar, M. (2017). Growing Pains: A Case Study for Large-Scale Vehicle Routing. *INFORMS Transactions on Education*, 17(2), 81–84. https://doi.org/10.1287/ited.2016.0167cs