"""
vrp_solvers/__init__.py — Public API for the vrp_solvers package.

Solver classes
--------------
ClarkeWrightSolver      Base CVRP — parallel savings construction + local search
NearestNeighborSolver   Base CVRP — nearest-neighbor construction + local search
ALNSSolver              Base CVRP — Adaptive Large Neighborhood Search
SimulatedAnnealingSolver Base CVRP — Simulated Annealing
TabuSearchSolver        Base CVRP — Tabu Search
MixedFleetSolver        Q2 — heterogeneous fleet (vans + straight trucks)
ALNSMixedFleetSolver    Q2 — ALNS variant for mixed fleet
OvernightSolver         Q3 — DOT overnight break wrapper over any base solver
ORToolsSolver           Q1/Q2/Q3 — Google OR-Tools CVRP (capacity + time windows)

Utility classes
---------------
CostModel               Weekly cost estimation (driver, fuel, fixed)
ResourceAnalyser        Minimum truck and driver requirements

Core helpers (re-exported from base)
-------------------------------------
loadInputs              Read orders and distance matrix from data/
evaluateRoute           Simulate a route and return feasibility + cost metrics
evaluateMixedRoute      Same as evaluateRoute but vehicle-type-aware
getDistance             Distance matrix lookup by ZIP pair
toClock                 Fractional hours → HH:MM string
routeIds                Extract ordered list of order IDs from a route
applyLocalSearch        2-opt + or-opt pass over a list of routes
consolidateRoutes       Repeatedly eliminate the smallest route if feasible
detailedRouteTrace      Per-stop timing rows for Table 3 CSV output

Overnight helpers (re-exported from overnightSolver)
-----------------------------------------------------
evaluateOvernightRoute  Evaluate a two-day paired route under DOT HOS
applyOvernightImprovements  Greedy overnight pairing pass over a weekly route set
"""

# Solver classes
from vrp_solvers.clarkeWright       import ClarkeWrightSolver
from vrp_solvers.nearestNeighbor    import NearestNeighborSolver
from vrp_solvers.alns               import ALNSSolver
from vrp_solvers.simulatedAnnealing import SimulatedAnnealingSolver
from vrp_solvers.tabuSearch         import TabuSearchSolver
from vrp_solvers.mixedFleetSolver   import MixedFleetSolver, ALNSMixedFleetSolver
from vrp_solvers.overnightSolver    import OvernightSolver
from vrp_solvers.orToolsSolver          import ORToolsSolver
from vrp_solvers.orToolsOvernightSolver import ORToolsOvernightSolver
from vrp_solvers.angularSweepAssigner   import AngularSweepAssigner
from vrp_solvers.cpSatAssigner          import CpSatAssigner

# Utility classes
from vrp_solvers.costModel        import CostModel
from vrp_solvers.resourceAnalyser import ResourceAnalyser

# Core helpers
from vrp_solvers.base import (
    loadInputs,
    evaluateRoute,
    evaluateMixedRoute,
    getDistance,
    toClock,
    routeIds,
    applyLocalSearch,
    consolidateRoutes,
    detailedRouteTrace,
)

# Overnight helpers
from vrp_solvers.overnightSolver import (
    evaluateOvernightRoute,
    applyOvernightImprovements,
)

__all__ = [
    # Solvers
    "ClarkeWrightSolver",
    "NearestNeighborSolver",
    "ALNSSolver",
    "SimulatedAnnealingSolver",
    "TabuSearchSolver",
    "MixedFleetSolver",
    "ALNSMixedFleetSolver",
    "OvernightSolver",
    "ORToolsSolver",
    "ORToolsOvernightSolver",
    "AngularSweepAssigner",
    "CpSatAssigner",
    # Utilities
    "CostModel",
    "ResourceAnalyser",
    # Core helpers
    "loadInputs",
    "evaluateRoute",
    "evaluateMixedRoute",
    "getDistance",
    "toClock",
    "routeIds",
    "applyLocalSearch",
    "consolidateRoutes",
    "detailedRouteTrace",
    # Overnight helpers
    "evaluateOvernightRoute",
    "applyOvernightImprovements",
]
