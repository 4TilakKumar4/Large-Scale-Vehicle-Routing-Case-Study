"""
vrp_solvers — VRP solution algorithm package for the NHG case study.
Run VRP_DataAnalysis.py first to populate data/ before importing any solver.
Usage:
    from vrp_solvers.base import loadInputs, DAYS
    from vrp_solvers.clarkeWright import ClarkeWrightSolver
    from vrp_solvers.tabuSearch import TabuSearchSolver
    orders, distMatrix = loadInputs()
    solver = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)
    routes = solver.solve(dayOrders)
"""

from vrp_solvers.base import loadInputs, evaluateRoute, DAYS
from vrp_solvers.clarkeWright        import ClarkeWrightSolver
from vrp_solvers.nearestNeighbor     import NearestNeighborSolver
from vrp_solvers.tabuSearch          import TabuSearchSolver
from vrp_solvers.simulatedAnnealing  import SimulatedAnnealingSolver
from vrp_solvers.alns                import ALNSSolver
from vrp_solvers.resourceAnalyser    import ResourceAnalyser
from vrp_solvers.costModel           import CostModel, SENSITIVITY_RANGES
from vrp_solvers.orToolsSolver       import ORToolsSolver
from vrp_solvers.relaxedScheduleSolver import (
    SweepRelaxedSolver,
    ALNSRelaxedSolver,
    getVisitGroups,
)
from vrp_solvers.mixedFleetSolver    import MixedFleetSolver, ALNSMixedFleetSolver
from vrp_solvers.overnightSolver     import (
    OvernightSolver,
    evaluateOvernightRoute,
    applyOvernightImprovements,
    findAllOvernightCandidates,
)

__all__ = [
    "loadInputs",
    "evaluateRoute",
    "DAYS",
    "ClarkeWrightSolver",
    "NearestNeighborSolver",
    "TabuSearchSolver",
    "SimulatedAnnealingSolver",
    "ALNSSolver",
    "ResourceAnalyser",
    "CostModel",
    "SENSITIVITY_RANGES",
    "ORToolsSolver",
    "SweepRelaxedSolver",
    "ALNSRelaxedSolver",
    "getVisitGroups",
    "MixedFleetSolver",
    "ALNSMixedFleetSolver",
    "OvernightSolver",
    "evaluateOvernightRoute",
    "applyOvernightImprovements",
    "findAllOvernightCandidates",
    "RelaxedScheduleSolver"
]