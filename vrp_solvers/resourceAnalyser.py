"""
vrp_solvers/resourceAnalyser.py — Minimum truck and driver requirements for a weekly route set.

Truck count  : one truck per simultaneous route; weekly peak = max daily route count.
Driver count : minimum path cover across the 5-day route graph.
               Solved as total_routes - max_bipartite_matching, where an edge exists
               between route A (day N) and route B (day N+1) when the driver finishing A
               satisfies the 10-hour DOT break before B's dispatch time.
               Overnight pairings are pre-committed chains and are resolved first.
"""

import os

from vrp_solvers.base import (
    BREAK_TIME,
    DAYS,
    DEPOT_ZIP,
    DRIVING_SPEED,
    WINDOW_OPEN,
    evaluateRoute,
    getDistance,
)

ADJACENT_DAYS = [
    ("Mon", "Tue"),
    ("Tue", "Wed"),
    ("Wed", "Thu"),
    ("Thu", "Fri"),
]


class ResourceAnalyser:
    """
    Computes minimum trucks and drivers for a given weekly routing solution.
    Optionally accepts overnight pairings so pre-committed driver chains are
    excluded from the general assignment problem.

    Usage:
        analyser = ResourceAnalyser(routesByDay)
        analyser.analyse()
        report = analyser.getReport()
    """

    def __init__(self, routesByDay, overnightPairings=None):
        self.routesByDay       = routesByDay
        self.overnightPairings = overnightPairings or []
        self._trucksByDay      = {}
        self._minDrivers       = None
        self._driverChains     = []
        self._analysed         = False

    def analyse(self):
        """Run truck and driver calculations. Must be called before getReport()."""
        self._trucksByDay  = self._computeTrucks()
        self._minDrivers, self._driverChains = self._computeDrivers()
        self._analysed = True

    def getReport(self):
        """Return a dict of resource metrics from the last analyse() call."""
        if not self._analysed:
            raise RuntimeError("Call analyse() before getReport().")

        totalRoutes  = sum(len(r) for r in self.routesByDay.values())
        overnightCount = len(self.overnightPairings)

        return {
            "min_drivers":          self._minDrivers,
            "min_trucks_peak":      max(self._trucksByDay.values(), default=0),
            "trucks_by_day":        dict(self._trucksByDay),
            "total_routes":         totalRoutes,
            "overnight_pairs":      overnightCount,
            "day_cab_routes":       totalRoutes - overnightCount * 2,
        }

    def printReport(self):
        """Print a formatted resource summary to stdout."""
        if not self._analysed:
            raise RuntimeError("Call analyse() before printReport().")

        report = self.getReport()
        print("\nResource Requirements")
        print("-" * 60)
        print(f"  Minimum drivers needed:   {report['min_drivers']}")
        print(f"  Peak trucks (any one day): {report['min_trucks_peak']}")
        print(f"  Total routes this week:   {report['total_routes']}")
        print(f"  Overnight pairings:        {report['overnight_pairs']}")

        print("\n  Trucks by day:")
        for day, count in report["trucks_by_day"].items():
            print(f"    {day}: {count} truck(s)")

        print("\n  Driver chains:")
        for i, chain in enumerate(self._driverChains, start=1):
            chainStr = " → ".join(f"{day}[R{ridx + 1}]" for day, ridx in chain)
            print(f"    Driver {i:2d}: {chainStr}")

    def _computeTrucks(self):
        """
        One truck per route per day — trucks cannot be reused intra-day because
        MAD loads trailers in advance and routes run concurrently.
        """
        return {day: len(routes) for day, routes in self.routesByDay.items()}

    def _computeDrivers(self):
        """
        Minimum path cover on the 5-day route DAG.

        Nodes  : every (day, route_index) pair.
        Edges  : (day_N, i) → (day_N+1, j) when the driver finishing route i
                 satisfies the 10-hour break before route j's dispatch time.
        Result : total_nodes - max_bipartite_matching.

        Overnight pairings are pre-committed chains removed before matching.
        """
        # --- collect per-route timing ---
        routeTiming = {}   # (day, ridx) → {"dispatch": float, "return": float}
        for day, routes in self.routesByDay.items():
            for ridx, route in enumerate(routes):
                if not route:
                    continue
                dispatchTime = self._dispatchTime(route)
                returnTime   = evaluateRoute(route)["return_time"]
                routeTiming[(day, ridx)] = {
                    "dispatch": dispatchTime,
                    "return":   returnTime,
                }

        # --- mark overnight-consumed routes ---
        # overnight driver covers both day1 and day2 routes; neither enters matching
        overnightConsumed = set()
        for pairing in self.overnightPairings:
            d1, r1 = pairing["day1"], pairing["route1_idx"]
            d2, r2 = pairing["day2"], pairing["route2_idx"]
            overnightConsumed.add((d1, r1))
            overnightConsumed.add((d2, r2))

        # --- build eligibility edges for adjacent-day pairs ---
        # edges[left_node] = list of right_nodes the same driver could cover next
        edges = {}
        for d1, d2 in ADJACENT_DAYS:
            leftRoutes  = [
                (d1, ridx) for ridx in range(len(self.routesByDay.get(d1, [])))
                if (d1, ridx) not in overnightConsumed
            ]
            rightRoutes = [
                (d2, ridx) for ridx in range(len(self.routesByDay.get(d2, [])))
                if (d2, ridx) not in overnightConsumed
            ]

            for leftNode in leftRoutes:
                edges.setdefault(leftNode, [])
                leftReturn = routeTiming.get(leftNode, {}).get("return", float("inf"))

                for rightNode in rightRoutes:
                    rightDispatch = routeTiming.get(rightNode, {}).get("dispatch", float("inf"))
                    # Day 2 times are relative to midnight of day 2, so add 24 hours
                    # to compare against a day 1 return time correctly
                    rightDispatchAbsolute = rightDispatch + 24.0
                    earliestAvailable     = leftReturn + BREAK_TIME

                    if earliestAvailable <= rightDispatchAbsolute:
                        edges[leftNode].append(rightNode)

        # --- max bipartite matching via augmenting paths ---
        allNodes = [
            (day, ridx)
            for day in DAYS
            for ridx in range(len(self.routesByDay.get(day, [])))
            if (day, ridx) not in overnightConsumed
        ]

        matchLeft  = {}   # left_node  → matched right_node
        matchRight = {}   # right_node → matched left_node

        def augment(node, visited):
            for candidate in edges.get(node, []):
                if candidate in visited:
                    continue
                visited.add(candidate)
                if candidate not in matchRight or augment(matchRight[candidate], visited):
                    matchLeft[node]       = candidate
                    matchRight[candidate] = node
                    return True
            return False

        for node in allNodes:
            # Only left-side nodes (not the last day) initiate augmentation
            if node[0] != DAYS[-1]:
                augment(node, set())

        matchingSize = len(matchLeft)

        # --- reconstruct driver chains ---
        # start from unmatched left-side nodes and follow the match chain forward
        chains        = []
        chainedNodes  = set()

        # overnight chains first
        for pairing in self.overnightPairings:
            d1, r1 = pairing["day1"], pairing["route1_idx"]
            d2, r2 = pairing["day2"], pairing["route2_idx"]
            chains.append([(d1, r1), (d2, r2)])
            chainedNodes.add((d1, r1))
            chainedNodes.add((d2, r2))

        # day-cab chains from matching
        for node in allNodes:
            if node in chainedNodes:
                continue
            # only start a chain from a node that has no predecessor in the matching
            if node not in matchRight:
                chain   = [node]
                current = node
                chainedNodes.add(current)
                while current in matchLeft:
                    nxt = matchLeft[current]
                    chain.append(nxt)
                    chainedNodes.add(nxt)
                    current = nxt
                chains.append(chain)

        # any remaining nodes not captured above get their own single-day chain
        for node in allNodes:
            if node not in chainedNodes:
                chains.append([node])
                chainedNodes.add(node)

        totalNonOvernight = len(allNodes)
        minDrivers        = len(self.overnightPairings) + (totalNonOvernight - matchingSize)

        return minDrivers, chains

    def _dispatchTime(self, route):
        """Recompute dispatch time from the first stop — mirrors the logic in evaluateRoute."""
        if not route:
            return float(WINDOW_OPEN)
        firstZip   = route[0]["TOZIP"]
        firstDrive = getDistance(DEPOT_ZIP, firstZip) / DRIVING_SPEED
        return max(0.0, WINDOW_OPEN - firstDrive)