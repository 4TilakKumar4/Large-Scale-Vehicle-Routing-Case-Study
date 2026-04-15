# Large-Scale Vehicle Routing: NHG Case Study
### IE 7200 — Supply Chain Engineering | Spring 2026

---

## Overview

This project implements and compares multiple solution methodologies for a large-scale Capacitated Vehicle Routing Problem (CVRP) with time windows and DOT Hours-of-Service constraints. The problem is drawn from the *Growing Pains* teaching case (Milburn, Kirac, and Hadianniasar, 2017), set in the context of Northeastern Home Goods (NHG) and their proposed carrier Massachusetts Area Distribution (MAD), operating out of a single distribution center in Wilmington, Massachusetts.

The objective is to construct a minimum-cost set of weekly delivery routes serving 123 store locations across six northeastern states, subject to vehicle capacity, delivery time windows, and federal DOT drive and duty time regulations.

---

## Documentation

| Document | Description |
|---|---|
| [Methodology](docs/METHODOLOGY.md) | Algorithm design, problem formulation, implementation details |
| [Results](docs/RESULTS.md) | Comparison tables, findings, and interpretation |
| [Data Description](docs/DATA_DESCRIPTION.md) | Dataset structure, column definitions, input statistics |

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
│   ├── alns.py                      ← ALNSSolver
│   ├── overnightSolver.py           ← OvernightSolver + overnight evaluation logic
│   └── resourceAnalyser.py          ← ResourceAnalyser (trucks + drivers)
│
├── tests/                           ← Unit tests (123 tests, no external dependencies)
│   ├── __init__.py
│   ├── conftest.py
│   ├── fixtures.py                  ← Synthetic dataset injected into base module
│   ├── test_base.py
│   ├── test_clarkeWright.py
│   ├── test_nearestNeighbor.py
│   ├── test_tabuSearch.py
│   ├── test_simulatedAnnealing.py
│   ├── test_alns.py
│   ├── test_overnightSolver.py
│   └── test_resourceAnalyser.py
│
├── docs/
│   ├── METHODOLOGY.md
│   ├── RESULTS.md
│   └── DATA_DESCRIPTION.md
│
├── data/                            ← Produced by VRP_DataAnalysis.py
│   ├── orders_clean.csv
│   ├── locations_clean.csv
│   └── distance_matrix.csv
│
└── outputs/                         ← All generated plots and CSVs
    ├── eda/
    ├── comparison/
    └── routes_map.html
```

---

## Run Order

`VRP_DataAnalysis.py` must be executed first. It reads the raw Excel inputs, cleans them, and writes processed CSVs to `data/`. Every downstream script reads from `data/` — none access the raw Excel files directly.

```
deliveries.xlsx ──┐
distances.xlsx  ──┴──► VRP_DataAnalysis.py ──► data/
                                                  │
                        ┌─────────────────────────┘
                        ▼
                   vrp_solvers/base.py  (loadInputs reads from data/)
                        │
           ┌────────────┼──────────────────────┬─────────────────────┐
           ▼            ▼                      ▼                     ▼
    VRP_BaseCase   VRP_BaseCase_Map   VRP_SolverComparison   VRP_OvernightRoutes
           │            │                      │                     │
           ▼            ▼                      ▼                     ▼
      (console)  outputs/routes_map.html  outputs/comparison/   (console)
```

```bash
python VRP_DataAnalysis.py        # always run first
python VRP_BaseCase.py            # base case routes and console report
python VRP_BaseCase_Map.py        # base case + interactive HTML map
python VRP_SolverComparison.py    # all 9 algorithm configurations (~10-20 min)
python VRP_OvernightRoutes.py     # overnight scenario, independent of above
```

> All scripts must be run from the project root. The `vrp_solvers` package is resolved relative to the working directory.

---

## Running Tests

```bash
python -m pytest tests/ -v                   # verbose — one line per test
python -m pytest tests/ -v -s                # also shows print() output (useful for seeing solver warnings)
python -m pytest tests/test_alns.py -v       # run just one file
python -m pytest tests/ -k "ALNS" -v         # run only tests whose name contains "ALNS"
```

Tests use a synthetic minimal dataset and do not require `data/` to exist. All 123 tests pass on a clean clone before `VRP_DataAnalysis.py` has been run.

---

## Dependencies

```bash
pip install -r requirements.txt
```

---

## Reference

Milburn, A.B., Kirac, E., and Hadianniasar, M. (2017). Growing Pains: A Case Study for Large-Scale Vehicle Routing. *INFORMS Transactions on Education*, 17(2), 81–84. https://doi.org/10.1287/ited.2016.0167cs