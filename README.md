# Large-Scale Vehicle Routing: NHG Case Study
### IE 7200 — Supply Chain Engineering | Spring 2026

## Overview

This project implements and compares multiple solution methodologies for a large-scale Capacitated Vehicle Routing Problem (CVRP) with time windows and DOT Hours-of-Service constraints. The problem is drawn from the *Growing Pains* teaching case (Milburn, Kirac, and Hadianniasar, 2017), set in the context of Northeastern Home Goods (NHG) and their proposed carrier Massachusetts Area Distribution (MAD), operating out of a single distribution center in Wilmington, Massachusetts.

The objective is to construct a minimum-cost set of weekly delivery routes serving 123 store locations across six northeastern states, subject to vehicle capacity, delivery time windows, and federal DOT drive and duty time regulations.

## Documentation

| Document | Description |
|---|---|
| [Methodology](docs/METHODOLOGY.md) | Algorithm design, problem formulation, cost model, sensitivity analysis |
| [Results](docs/RESULTS.md) | Comparison tables, findings, and interpretation |
| [Data Description](docs/DATA_DESCRIPTION.md) | Dataset structure, column definitions, input statistics |

## Repository Structure

```
project/
│
├── deliveries.xlsx                  ← Raw input: order table and location table
├── distances.xlsx                   ← Raw input: 124×124 road distance matrix (miles)
│
├── VRP_DataAnalysis.py              ← Phase 1: data cleaning, EDA, and CSV export
│
├── VRP_BaseCase.py                  ← Sub-problem 1: base case solver (CW + LS) + Folium map
├── VRP_OvernightRoutes.py           ← Sub-problem 1 extension: overnight DOT break + Folium map
├── VRP_MixedFleet.py                ← Sub-problem 2: mixed fleet (Van + ST) + Folium map
│                                       All three accept --no-map to skip geocoding
│
├── VRP_SolverComparison.py          ← Ten-algorithm comparison across all sub-problems
├── VRP_CostAnalysis.py              ← Cost estimation and cost-rate sensitivity analysis
├── VRP_SensitivityAnalysis.py       ← Operational sensitivity analysis (demand, fleet, DOT)
│
├── vrp_solvers/                     ← Solver package
│   ├── __init__.py
│   ├── base.py                      ← Constants, data I/O, route evaluation, local search
│   ├── clarkeWright.py              ← ClarkeWrightSolver
│   ├── nearestNeighbor.py           ← NearestNeighborSolver
│   ├── tabuSearch.py                ← TabuSearchSolver
│   ├── simulatedAnnealing.py        ← SimulatedAnnealingSolver
│   ├── alns.py                      ← ALNSSolver
│   ├── overnightSolver.py           ← OvernightSolver + overnight evaluation logic
│   ├── mixedFleetSolver.py          ← MixedFleetSolver (Van + ST fleet assignment)
│   ├── resourceAnalyser.py          ← ResourceAnalyser (trucks + drivers via min path cover)
│   └── costModel.py                 ← CostModel (6-component cost estimation, 2024-25 rates)
│
├── tests/
│   ├── fixtures.py                  ← Synthetic 7-ZIP dataset injected into base module
│   ├── test_base.py
│   ├── test_clarkeWright.py
│   ├── test_nearestNeighbor.py
│   ├── test_tabuSearch.py
│   ├── test_simulatedAnnealing.py
│   ├── test_alns.py
│   ├── test_overnightSolver.py
│   ├── test_resourceAnalyser.py
│   ├── test_mixedFleetSolver.py
│   ├── test_costModel.py
│   └── test_constantOverride.py
│
├── docs/
│   ├── METHODOLOGY.md
│   ├── RESULTS.md
│   └── DATA_DESCRIPTION.md
│
├── data/                            ← Produced by VRP_DataAnalysis.py (not committed)
│   ├── orders_clean.csv
│   ├── locations_clean.csv
│   └── distance_matrix.csv
│
└── outputs/                         ← All generated plots and CSVs (not committed)
    ├── eda/
    ├── base_case/          ← route_details.csv, resource_summary.csv, driver_chains.csv,
    │                          routes_map_baseCase.html (when --no-map not passed)
    ├── overnight/          ← same + routes_map_overnight.html
    ├── mixed_fleet/        ← same + fleet_analysis.png, route_map.png,
    │                          routes_map_mixed_fleet.html
    ├── comparison/
    ├── cost_analysis/
    └── sensitivity/
```

## Run Order

`VRP_DataAnalysis.py` must be executed first. It reads the raw Excel inputs, cleans them, and writes processed CSVs to `data/`. Every downstream script reads from `data/`.

```
deliveries.xlsx ──┐
distances.xlsx  ──┴──► VRP_DataAnalysis.py ──► data/
                                                  │
                        ┌─────────────────────────┘
                        ▼
                   vrp_solvers/base.py  (loadInputs reads from data/)
                        │
          ┌─────────────┼──────────────────┬──────────────────┐
          ▼             ▼                  ▼                  ▼
   VRP_BaseCase   VRP_MixedFleet   VRP_SolverComparison   VRP_OvernightRoutes
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                     VRP_CostAnalysis      VRP_SensitivityAnalysis
```

```bash
python VRP_DataAnalysis.py                  # always run first
python VRP_BaseCase.py --no-map             # fast solver run (no geocoding)
python VRP_BaseCase.py                      # solver + interactive Folium map
python VRP_OvernightRoutes.py --no-map      # fast overnight solver run
python VRP_OvernightRoutes.py               # overnight solver + interactive map
python VRP_MixedFleet.py --no-map           # fast mixed fleet run
python VRP_MixedFleet.py                    # mixed fleet + interactive map
python VRP_SolverComparison.py              # all 12 algorithm configurations (~20-30 min)
python VRP_CostAnalysis.py                  # cost breakdown and sensitivity (~25 min)
python VRP_SensitivityAnalysis.py           # operational sensitivity analysis (~35 min)
```

All scripts must be run from the project root. The `--no-map` flag skips geocoding and
Folium map generation — useful during solver iteration when you only need console output.

## Running Tests

```bash
python -m pytest tests/ -v                   # verbose — one line per test
python -m pytest tests/ -v -s                # also shows print() output
python -m pytest tests/test_costModel.py -v  # run just one file
python -m pytest tests/ -k "Mixed" -v        # run only tests matching pattern
```

Tests use a synthetic minimal dataset injected directly into `vrp_solvers.base` and do not require `data/` to exist. All 205 tests pass on a clean clone before `VRP_DataAnalysis.py` has been run.

## Dependencies

```bash
pip install -r requirements.txt
```

## Reference

Milburn, A.B., Kirac, E., and Hadianniasar, M. (2017). Growing Pains: A Case Study for Large-Scale Vehicle Routing. *INFORMS Transactions on Education*, 17(2), 81–84. https://doi.org/10.1287/ited.2016.0167cs