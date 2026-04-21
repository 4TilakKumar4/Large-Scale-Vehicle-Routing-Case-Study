"""
tests/fixtures.py — Synthetic minimal dataset for unit tests.

Defines a depot and six store ZIPs with hand-crafted distances and order records.
Injects the distance matrix directly into vrp_solvers.base so tests run without
needing data/ files or VRP_DataAnalysis.py to have been executed first.

Distances are in miles and chosen so that:
  - Each single-stop route is feasible (cube < VAN_CAPACITY, drive < MAX_DRIVING)
  - A two-stop route combining stores A+B is feasible (combined cube still < VAN_CAPACITY)
  - Stores on Mon vs Tue allow overnight pairing tests

Topology (all from depot ZIP 1887):
  ZIP 1001 — 10 miles from depot,  5 miles to ZIP 1002
  ZIP 1002 — 12 miles from depot,  5 miles to ZIP 1001
  ZIP 1003 — 20 miles from depot, 15 miles to ZIP 1004
  ZIP 1004 — 22 miles from depot, 15 miles to ZIP 1003
  ZIP 1005 — 30 miles from depot, 25 miles to ZIP 1006
  ZIP 1006 — 32 miles from depot, 25 miles to ZIP 1005
"""

import pandas as pd
import vrp_solvers.base as base

DEPOT = 1887

ALL_ZIPS = [DEPOT, 1001, 1002, 1003, 1004, 1005, 1006]

# fmt: off
RAW_DISTANCES = {
    DEPOT: {DEPOT:  0, 1001: 10, 1002: 12, 1003: 20, 1004: 22, 1005: 30, 1006: 32},
    1001:  {DEPOT: 10, 1001:  0, 1002:  5, 1003: 18, 1004: 20, 1005: 28, 1006: 30},
    1002:  {DEPOT: 12, 1001:  5, 1002:  0, 1003: 16, 1004: 18, 1005: 26, 1006: 28},
    1003:  {DEPOT: 20, 1001: 18, 1002: 16, 1003:  0, 1004:  5, 1005: 15, 1006: 17},
    1004:  {DEPOT: 22, 1001: 20, 1002: 18, 1003:  5, 1004:  0, 1005: 13, 1006: 15},
    1005:  {DEPOT: 30, 1001: 28, 1002: 26, 1003: 15, 1004: 13, 1005:  0, 1006:  5},
    1006:  {DEPOT: 32, 1001: 30, 1002: 28, 1003: 17, 1004: 15, 1005:  5, 1006:  0},
}
# fmt: on


def buildDistMatrix():
    """Return a pandas DataFrame distance matrix over ALL_ZIPS."""
    return pd.DataFrame(RAW_DISTANCES, index=ALL_ZIPS, columns=ALL_ZIPS).astype(float)


def makeOrder(orderId, toZip, cube, day):
    """Return a single order dict matching the structure evaluateRoute expects."""
    return {
        "ORDERID":    orderId,
        "FROMZIP":    DEPOT,
        "TOZIP":      toZip,
        "CUBE":       float(cube),
        "DayOfWeek":  day,
        "straight_truck_required": "no",
    }


def makeOrdersDf(orders):
    """Wrap a list of order dicts into a DataFrame."""
    return pd.DataFrame(orders)


# Six orders: three on Mon, three on Tue — cubes chosen so pairs fit in one van
MON_ORDERS = [
    makeOrder(1, 1001, 500,  "Mon"),
    makeOrder(2, 1002, 600,  "Mon"),
    makeOrder(3, 1003, 700,  "Mon"),
]

TUE_ORDERS = [
    makeOrder(4, 1004, 400,  "Tue"),
    makeOrder(5, 1005, 450,  "Tue"),
    makeOrder(6, 1006, 500,  "Tue"),
]

ALL_ORDERS = MON_ORDERS + TUE_ORDERS

MON_DF  = makeOrdersDf(MON_ORDERS)
TUE_DF  = makeOrdersDf(TUE_ORDERS)
ALL_DF  = makeOrdersDf(ALL_ORDERS)


# ---------------------------------------------------------------------------
# Mixed fleet orders — for test_mixedFleetSolver.py
# ---------------------------------------------------------------------------

# ST-required orders — must go on Straight Truck
ST_ORDERS = [
    {**makeOrder(10, 1001, 300, "Mon"), "straight_truck_required": "yes"},
    {**makeOrder(11, 1002, 400, "Mon"), "straight_truck_required": "yes"},
]

# Over-capacity for ST (cube > ST_CAPACITY 1400) — must go on Van
LARGE_ORDERS = [
    makeOrder(20, 1003, 1500, "Mon"),   # cube 1500 > ST_CAP 1400
]

# Flexible orders — can go on either fleet
FLEX_ORDERS = [
    makeOrder(30, 1004, 300, "Mon"),
    makeOrder(31, 1005, 350, "Mon"),
]

# Full mixed day: 2 ST-required + 1 too-large + 2 flexible
MIXED_MON_ORDERS = ST_ORDERS + LARGE_ORDERS + FLEX_ORDERS
MIXED_MON_DF     = makeOrdersDf(MIXED_MON_ORDERS)


def injectFixture():
    """
    Inject the synthetic distance matrix into vrp_solvers.base so all solver
    imports resolve correctly without reading any files from disk.
    Must be called at the top of every test module before importing solvers.
    """
    base.DIST_MATRIX = buildDistMatrix()
    base.ORDERS      = ALL_DF
