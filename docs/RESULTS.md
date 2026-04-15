# Methodology
### IE 7200 — Supply Chain Engineering | NHG Vehicle Routing Case Study

## 1. Problem Formulation

### 1.1 Problem Classification

The NHG routing problem is classified as a **Capacitated Vehicle Routing Problem with Time Windows and HOS Constraints (CVRPTW-HOS)**. It combines elements of three canonical VRP variants:

- **CVRP** — vehicles have a fixed volume capacity
- **VRPTW** — each customer must be served within a time window
- **PVRP** — deliveries follow a fixed weekly repeating schedule

The problem is solved independently for each weekday. Orders assigned to Monday are routed on Monday routes only; cross-day mixing is not permitted in the base case (see Section 6 for the overnight extension).

### 1.2 Formal Definition

**Sets:**
- $N$ — set of store orders (delivery stops)
- $K$ — set of available vehicles (homogeneous fleet of vans in Sub-problem 1; mixed fleet in Sub-problem 2)
- Depot located at ZIP code 01887 (Wilmington, MA)

**Parameters:**

| Symbol | Value | Description |
|---|---|---|
| $Q_V$ | 3,200 ft³ | Van capacity |
| $Q_{ST}$ | 1,400 ft³ | Straight Truck capacity (Sub-problem 2) |
| $r_V$ | 0.03 min/ft³ | Van unload rate |
| $r_{ST}$ | 0.043 min/ft³ | Straight Truck unload rate |
| $t_{min}$ | 30 min | Minimum unload time per stop |
| $v$ | 40 mph | Driving speed |
| $[e, l]$ | [08:00, 18:00] | Delivery time window |
| $H_d$ | 11 hours | Maximum driving per shift |
| $H_o$ | 14 hours | Maximum on-duty per shift |
| $H_b$ | 10 hours | Mandatory break duration |

**Decision variables:** Route assignment and stop sequencing for each vehicle on each day.

**Objective:** Minimise total weekly route miles.

### 1.3 Feasibility Constraints

A route is feasible if and only if all three of the following hold simultaneously:

**Capacity:** $\sum_{i \in \text{route}} \text{cube}_i \leq Q$ (where $Q = Q_V$ or $Q_{ST}$ depending on vehicle type)

**Time windows:** Every stop $i$ must begin and complete service within $[08:00, 18:00]$. Service start time is $\max(\text{arrival}_i, 08:00)$; service completes at start time plus unload time. The window check applies to service completion, not just arrival.

**DOT HOS:** Total driving $\leq H_d$ and total on-duty time $\leq H_o$, where on-duty includes driving, unloading, and waiting time. Loading at the DC does not count toward on-duty hours.

### 1.4 Dispatch Logic

Vehicles are dispatched at the time required to arrive at the first stop exactly at 08:00:

$$t_{\text{dispatch}} = \max\left(0, 08:00 - \frac{d(\text{depot}, \text{first stop})}{v}\right)$$

The floor of zero prevents dispatch before midnight. On-duty time begins accumulating from dispatch.

### 1.5 Unload Time

$$t_{\text{unload}}(q) = \frac{\max(t_{\min},\ r \cdot q)}{60} \text{ hours}$$

## 2. Route Evaluation

All route evaluation is centralised in `vrp_solvers/base.py::evaluateRoute()` for Van routes and `evaluateMixedRoute()` for fleet-typed evaluation. These functions are called millions of times during construction and improvement and are the computational bottleneck of every solver. Each simulates the full route from dispatch to depot return and returns:

- Total miles, drive time, unload time, wait time, duty time
- Cube loaded
- Return time
- Three individual feasibility flags (capacity, window, DOT) and an overall feasibility flag

Both functions are intentionally stateless — they take a route list and return a dict.

## 3. Construction Heuristics

Construction heuristics build a complete feasible solution from scratch. They are fast, deterministic, and provide the starting point for all improvement methods.

### 3.1 Clarke-Wright Savings Algorithm

**Source:** Clarke and Wright (1964). Implemented in `vrp_solvers/clarkeWright.py`.

The algorithm starts with the most wasteful possible plan — a separate depot-to-store-to-depot round trip for every order — and iteratively merges pairs of routes using the savings criterion:

$$s(i, j) = d(\text{depot}, i) + d(\text{depot}, j) - d(i, j)$$

This measures how many miles are saved by visiting $i$ and $j$ on the same route rather than two separate ones. All $\binom{n}{2}$ pairs are computed and sorted descending. Merges are accepted in order when the merged route passes all three feasibility constraints and both orders are at valid endpoints of their current routes.

Four merge orientations are attempted at each candidate pair to handle both forward and reversed adjacency.

**Complexity:** $O(n^2 \log n)$ for savings computation.

**Route consolidation:** After CW construction, a post-processing phase attempts to eliminate the route with the smallest cube by relocating each of its stops to the cheapest feasible position in any remaining route. This is repeated until no further elimination is possible, directly reducing vehicle count.

### 3.2 Nearest Neighbor Heuristic

**Source:** Classical VRP literature. Implemented in `vrp_solvers/nearestNeighbor.py`.

Starting from the depot, the algorithm repeatedly appends the closest unvisited store whose addition keeps the current route feasible. When no feasible extension exists, the current route is closed and a new one is opened from the depot. $O(n^2)$ per day.

## 4. Local Search Improvement

### 4.1 2-opt

**Source:** Lin (1965). Implemented in `vrp_solvers/base.py::twoOptRoute()`.

For every pair of edges $(i, i+1)$ and $(j, j+1)$ in a route, 2-opt considers reversing the segment between them. If the reversal is feasible and reduces total route miles, it is accepted. Repeats until 2-optimal. Applied to routes with 4 or more stops.

### 4.2 Or-opt

**Source:** Or (1976). Implemented in `vrp_solvers/base.py::orOptRoute()`.

Or-opt tries relocating chains of 1, 2, or 3 consecutive stops to every other position within the same route. A relocation is accepted if it is feasible and reduces total miles. Applied to routes with 2 or more stops.

### 4.3 Note on 3-opt

3-opt removes three edges and reconnects the resulting segments in one of seven possible configurations, exploring a richer neighbourhood than 2-opt at $O(n^3)$ per pass. For the NHG instance sizes (up to 63 orders on Thursday), this amounts to approximately 250,000 edge triplets per iteration across multiple passes — significantly slower than the combined 2-opt + Or-opt pipeline with little practical gain. Bräysy and Gendreau (2005) demonstrate that Or-opt, by relocating chains of one to three consecutive stops, captures the most valuable subset of 3-opt improving moves at a fraction of the computational cost. The 2-opt + Or-opt pipeline used here therefore approximates 3-opt quality without the cubic runtime penalty.

### 4.4 Combined local search pipeline

```
twoOptRoute()  →  orOptRoute()
```

Applied in `vrp_solvers/base.py::applyLocalSearch()`.

## 5. Metaheuristics

All three metaheuristics seed from the CW + consolidation + 2-opt + Or-opt solution.

### 5.1 Tabu Search

**Source:** Glover (1986); Gendreau, Hertz, and Laporte (1994). Implemented in `vrp_solvers/tabuSearch.py`.

Considers relocate and swap inter-route moves. The best non-tabu feasible move is accepted at each iteration — even if it worsens the current solution. A tabu list keyed by `(move_type, source_route_idx, source_position)` prevents cycling. The aspiration criterion overrides the tabu list when a move produces a new global best.

| Parameter | Default |
|---|---|
| `maxIter` | 300 |
| `tabuTenure` | 15 |

### 5.2 Simulated Annealing

**Source:** Kirkpatrick, Gelatt, and Vecchi (1983); Osman (1993). Implemented in `vrp_solvers/simulatedAnnealing.py`.

Proposes a single random stop relocation per iteration. Acceptance follows the Metropolis criterion: improving moves always accepted; worsening moves accepted with probability $\exp(-\Delta/T)$. Temperature decays geometrically from $T_0$ to $T_{\text{end}}$.

| Parameter | Default |
|---|---|
| `maxIter` | 2000 |
| `tempStart` | 500.0 |
| `tempEnd` | 1.0 |

### 5.3 Adaptive Large Neighborhood Search

**Source:** Shaw (1998) for LNS; Ropke and Pisinger (2006) for ALNS. Implemented in `vrp_solvers/alns.py`.

At each iteration a destroy operator removes ~20% of stops and a repair operator reinserts them. Destroy operators: random removal, worst removal, Shaw removal, route removal. Repair operators: greedy insertion, regret-2 insertion. Operator weights are updated every 50 iterations using performance scores (9 for new global best, 5 for improvement, 2 for accepted worse move). Selection uses roulette-wheel sampling.

| Parameter | Default |
|---|---|
| `maxIter` | 500 |
| `removeFrac` | 0.2 |

## 6. Overnight Route Extension

**Implemented in:** `vrp_solvers/overnightSolver.py`

The overnight extension allows a driver to travel toward the next day's first delivery stop after completing day-1 deliveries, take the mandatory 10-hour DOT break en route, and resume on day 2. This can reduce total miles when adjacent-day routes serve geographically proximate stores.

**Break placement — latest break rule:** After completing all day-1 deliveries, the driver travels toward the day-2 first stop until their HOS limit is reached, then takes the break. If the driver reaches the day-2 stop before HOS runs out, the break begins on arrival. In both cases, day-2 HOS resets to zero when the break ends.

**Pairing algorithm:** For every pair of adjacent-day routes $(r_1, r_2)$, the overnight combination is evaluated. If feasible and shorter than the two separate routes, it is recorded as a candidate. Candidates are sorted by savings and applied greedily — each route participates in at most one pairing.

Overnight routes span exactly two consecutive calendar days. Adjacent pairs only: Mon-Tue, Tue-Wed, Wed-Thu, Thu-Fri.

### 6.6 Business framing — cost vs savings

Overnight routing reduces mileage but introduces two costs absent from day-cab operations: the IRS per diem allowance ($80/overnight) and the sleeper cab equipment premium ($60/day above a day-cab). The net annual saving or cost is computed by `CostModel.overnightSummary()` as:

$$\text{Net saving} = (\text{miles saved/week} \times \text{cost per mile} \times 52) - (\text{per diem annual} + \text{sleeper premium annual})$$

A positive result means overnight routing is economically justified. A negative result means the per diem and sleeper costs outweigh the mileage reduction — a realistic outcome when overnight routes save only a small number of miles per pairing.

## 7. Mixed Fleet — Sub-problem 2

**Implemented in:** `vrp_solvers/mixedFleetSolver.py`

Sub-problem 2 introduces a heterogeneous fleet of Vans (3,200 ft³, unload rate 0.030 min/ft³) and Straight Trucks (1,400 ft³, unload rate 0.043 min/ft³).

### 7.1 Fleet assignment rules

Each order falls into exactly one category:

| Category | Condition | Assignment |
|---|---|---|
| ST-required | `straight_truck_required == "yes"` | Straight Truck only |
| Too large for ST | `cube > ST_CAPACITY` | Van only |
| Flexible | All other orders | Either fleet |

### 7.2 Algorithm

The MixedFleetSolver runs a unified Clarke-Wright savings across all orders for the day, building a combined initial solution. After construction, fleet assignment is finalised: ST-required orders are moved to ST routes, over-capacity orders are confirmed on Van routes, and flexible orders are assigned to whichever fleet minimises total miles. A cross-fleet Or-opt improvement phase then relocates stops between Van and ST routes wherever it reduces miles without violating fleet assignment rules.

### 7.3 Route evaluation

Van and ST routes are evaluated by `evaluateMixedRoute(route, vehicleType)` which uses the appropriate capacity and unload rate for the specified fleet type.

### 7.4 ALNS Mixed Fleet Solver

`ALNSMixedFleetSolver` extends the mixed fleet approach by replacing the CW + local search improvement phase with ALNS operating over a **unified tagged route list**. Each route is represented as a dict `{"type": "van"|"st", "stops": [...]}`. This lets the ALNS destroy/repair loop move flexible stops between fleets — the key improvement over the CW-based solver, which assigns fleet types during construction and only repositions flexible stops via a post-processing cross-fleet pass.

**Seed:** `MixedFleetSolver` (CW + local search + cross-fleet improvement) provides the initial solution. ALNS then refines it.

**Destroy operators:** The same four operators as the base `ALNSSolver` (random, worst, Shaw, route removal), applied to the unified tagged route list. Removed stops carry their fleet constraint category implicitly.

**Repair operators (fleet-aware):** Each removed stop is classified before reinsertion:

| Category | Reinsertion rule |
|---|---|
| ST-required | Try ST routes only; open new ST route if no feasible position exists |
| Too large for ST | Try Van routes only; open new Van route if no feasible position exists |
| Flexible | Try all routes of both fleet types; insert at cheapest feasible position |

The flexible stop rule is the mechanism that enables genuine cross-fleet reallocation at each iteration — a stop that was on a Van in the seed may migrate to an ST route if the repair operator finds a cheaper feasible position there. This is not possible in `MixedFleetSolver` where fleet assignment is finalised during construction.

**Parameters:** `maxIter=500`, `removeFrac=0.2`, `randomSeed=42` — matching the base `ALNSSolver` for fair comparison.

## 8. Resource Analysis

**Implemented in:** `vrp_solvers/resourceAnalyser.py`

### 8.1 Truck requirements

One truck is required per simultaneous route. The weekly fleet requirement equals the maximum daily route count across all five weekdays.

### 8.2 Driver requirements — minimum path cover

The minimum driver count is solved as a minimum path cover on a directed acyclic graph. Nodes are (day, route_index) pairs. A directed edge from route $A$ on day $N$ to route $B$ on day $N+1$ exists when the driver finishing $A$ satisfies the 10-hour break requirement:

$$t_{\text{return}}(A) + 10 \leq t_{\text{dispatch}}(B) + 24$$

By König's theorem:

$$\text{min drivers} = \text{total routes} - \text{max bipartite matching}$$

Overnight pairings are pre-committed chains removed from the matching problem before solving, then added back as single driver assignments.

### 8.3 Driver workload sustainability

Beyond the minimum headcount, the `ResourceAnalyser` computes two workload metrics across all driver chains:

- **`avg_weekly_duty_hrs`** — average total on-duty hours per driver across their assigned routes for the week. A sustainable value for dedicated regional drivers is typically 40–55 hours/week.
- **`max_weekly_duty_hrs`** — the highest weekly duty load assigned to any single driver. If this approaches or exceeds 65–70 hours, the solution may minimise headcount at the expense of driver retention and turnover risk, which is a direct concern for NHG given their stated interest in operational stability.

The driver chains output (`outputs/*/driver_chains.csv`) lists each driver's full weekly assignment and per-day average duty hours, enabling NHG to evaluate whether the solution produces a sustainable and equitable work schedule.

### 8.4 DOT 70-hour/8-day rule

The DOT 70-hour rule is not modelled. The dataset represents a single average week; inter-week driver continuity is outside the project scope.

## 9. Cost Model

**Implemented in:** `vrp_solvers/costModel.py`

All default rates are sourced from 2024-25 industry benchmarks for Northeast US dedicated contract carriers.

### 9.1 Cost components

| Component | Basis | Default rate | Source |
|---|---|---|---|
| Mileage | Per mile driven | $0.725/mi (Van), $0.820/mi (ST) | ATRI 2025 operational costs report (2024 data) |
| Driver wages | Per on-duty hour; 1.5× above 8h | $32.00/hr base | BLS + Massachusetts ZipRecruiter 2024-25 |
| Benefits | Fraction of gross wages | 30% loading | ATRI benefits/wages ratio 24.7% + NE premium |
| Equipment | Per vehicle dispatched per day | $185/day (day-cab), +$60 sleeper, +$25 ST trailer | ATRI $0.39/mi + MA rental market data |
| Insurance | Per vehicle per operating day | $56/day | NE interstate fleet $15K/yr ÷ 268 operating days |
| Per diem | Per overnight pairing | $80/overnight | IRS Notice 2024-68, effective October 1 2024 |

Mileage cost covers fuel, repair and maintenance, and tyres. Equipment cost covers tractor and trailer lease/depreciation. Insurance covers $1M CSL liability, cargo, and physical damage as required for NE interstate operations. Per diem covers meals and incidental expenses for drivers on overnight routes.

### 9.2 Usage

```python
cm      = CostModel()                        # default 2024-25 NE rates
bd      = cm.weeklyBreakdown(routesByDay)    # per-route and weekly totals
annual  = cm.annualCost(bd["weekly"]["total"])
cm.printSummary(bd, label="Base Case")
```

Custom rates can be passed at construction time:

```python
cm = CostModel(driver_hourly_wage=35.00, cost_per_mile_van=0.80)
```

This allows NHG to substitute MAD's actual quoted rates and immediately compare against the internal estimate.

### 9.3 Cost model as a post-processing layer

The solvers in this project minimise total weekly miles as a proxy for cost. Miles and cost are highly correlated for routes of similar length, so this approximation is reasonable for comparing algorithm configurations. A cost-aware objective — minimising the full `CostModel` evaluation at each solver iteration — would be more precise but is computationally prohibitive given that cost evaluation involves per-route duty-hour accounting, overtime detection, and equipment classification. For this instance scale, the correlation between miles and cost is sufficiently tight that a miles-optimal solution is expected to be near cost-optimal. The `VRP_CostAnalysis.py` script verifies this post-hoc by applying the cost model to all ten algorithm configurations.

## 10. Sensitivity Analysis

### 10.1 Cost-rate sensitivity

**Implemented in:** `VRP_CostAnalysis.py`

Varies each of the six cost parameters (mileage rate, driver wage, overtime multiplier, overnight allowance, equipment daily cost, insurance daily cost) across a calibrated low-default-high range. Routes are not re-solved — only the cost calculation changes. Produces sensitivity curves for all five algorithm configurations simultaneously.

### 10.2 Operational sensitivity

**Implemented in:** `VRP_SensitivityAnalysis.py`

Ten operational sensitivities across three groups. Routes are re-solved for each parameter value.

**Group A — Demand:**

| Sensitivity | Parameter range |
|---|---|
| Volume scaling | +10%, +20%, +30% on all order cubes |
| Peak week | Wed + Thu orders scaled +25% |
| ST mix shift | Current fraction → 30% → 50% ST-required orders |

**Group B — Fleet and operational:**

| Sensitivity | Parameter range |
|---|---|
| Van capacity | 2,400 / 3,200 / 3,600 ft³ |
| ST capacity | 1,000 / 1,400 / 1,800 ft³ |
| Driving speed | 32 / 40 / 48 mph (±20%) |
| Unload rate | −20% / baseline / +20% (Van and ST simultaneously) |

**Group C — DOT and schedule:**

| Sensitivity | Parameter range |
|---|---|
| HOS safety buffer | 0h / 0.5h / 1h subtracted from MAX_DRIVING and MAX_DUTY |
| Delivery window | [08:00, 18:00] / [06:00, 20:00] / [00:00, 24:00] |
| Overnight toggle | No overnight vs overnight-allowed (cost delta) |

### 10.3 ConstantOverride — safe parameter patching

All Group B and C sensitivities require modifying the constants in `vrp_solvers/base.py` at runtime. This is done via the `ConstantOverride` context manager implemented in `VRP_SensitivityAnalysis.py`:

```python
with ConstantOverride(DRIVING_SPEED=32, MAX_DRIVING=10):
    routes = solver.solve(dayOrders)
# All constants guaranteed restored here, even if an exception occurred
```

The context manager patches module-level attributes directly on the `vrp_solvers.base` module object, which all solver functions read at call time. Source files are never modified. Restoration is guaranteed by a `finally` block — the original values are saved before any patch is applied and are always written back on exit regardless of whether the body succeeded or raised.

## 11. Algorithm Comparison Matrix

| Configuration | Construction | Local Search | Metaheuristic | Overnight | Fleet |
|---|---|---|---|---|---|
| `cw_only` | Clarke-Wright | — | — | — | Van |
| `nn_only` | Nearest Neighbor | — | — | — | Van |
| `cw_2opt_oropt` | Clarke-Wright | 2-opt + Or-opt | — | — | Van |
| `nn_2opt_oropt` | Nearest Neighbor | 2-opt + Or-opt | — | — | Van |
| `tabu_search` | CW | 2-opt + Or-opt (seed) | Tabu Search | — | Van |
| `simulated_annealing` | CW | 2-opt + Or-opt (seed) | Simulated Annealing | — | Van |
| `alns` | CW | 2-opt + Or-opt (seed) | ALNS | — | Van |
| `cw_overnight` | CW | 2-opt + Or-opt | — | Yes | Van |
| `alns_overnight` | CW | 2-opt + Or-opt (seed) | ALNS | Yes | Van |
| `mixed_fleet` | CW | 2-opt + Or-opt | — | — | Van + ST |
| `alns_mixed_fleet` | CW | 2-opt + Or-opt (seed) | ALNS | — | Van + ST |

> **Note on 3-opt:** 3-opt is not included as a standalone configuration. Or-opt (Section 4.2) captures the most computationally valuable subset of 3-opt improving moves at $O(n^2)$ per pass rather than $O(n^3)$, making the 2-opt + Or-opt pipeline the preferred choice for this instance scale (Bräysy and Gendreau, 2005).

## 12. Software Architecture

### 12.1 Package structure

All solver logic lives in `vrp_solvers/`. Each algorithm class exposes a consistent interface:

```python
solver = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)
routes = solver.solve(dayOrders)
stats  = solver.getStats()
curve  = solver.getConvergence()
```

`MixedFleetSolver` exposes additional fleet-specific accessors:

```python
solver = MixedFleetSolver()
solver.solve(dayOrders)
vanRoutes = solver.getVanRoutes()
stRoutes  = solver.getStRoutes()
stats     = solver.getStats()   # includes van_routes, st_routes, van_miles, st_miles
```

`OvernightSolver` wraps any base solver and operates on full weekly orders:

```python
solver = OvernightSolver(ALNSSolver())
routesByDay, overnightRoutes, usedRoutes = solver.solve(orders)
```

`ResourceAnalyser` takes a `routesByDay` dict and optional overnight pairings:

```python
analyser = ResourceAnalyser(routesByDay, overnightPairings=overnightRoutes)
analyser.analyse()
report = analyser.getReport()
```

`CostModel` is fully parameterised and works across all three sub-problems:

```python
cm = CostModel()
bd = cm.weeklyBreakdown(routesByDay)                                   # base case
bd = cm.weeklyBreakdown(routesByDay, overnightPairings=pairings)       # overnight
bd = cm.weeklyBreakdown(routesByDay=None, vanByDay=v, stByDay=st)      # mixed fleet
```

### 12.2 Shared base module

`vrp_solvers/base.py` is the single source of truth for all constants, data loading, and shared computation. The distance matrix is stored as a module-level global (`DIST_MATRIX`) populated by `loadInputs()`. All constants are defined once and read by all solvers at call time — enabling the `ConstantOverride` pattern used in sensitivity analysis without modifying source files.

### 12.3 Data flow

```
deliveries.xlsx  ──┐
distances.xlsx   ──┴──► VRP_DataAnalysis.py ──► data/
                                                   │
                         ┌─────────────────────────┘
                         ▼
                    vrp_solvers/base.py (loadInputs)
                         │
          ┌──────────────┼────────────────────────────┐
          ▼              ▼                            ▼
   VRP_BaseCase    VRP_MixedFleet            VRP_SolverComparison
   VRP_OvernightRoutes                               │
                                         ┌───────────┴───────────┐
                                         ▼                       ▼
                                VRP_CostAnalysis    VRP_SensitivityAnalysis
                                         │                       │
                                         ▼                       ▼
                                 outputs/cost_analysis/   outputs/sensitivity/
```

## 13. References

1. Clarke, G. and Wright, J.W. (1964). Scheduling of Vehicles from a Central Depot to a Number of Delivery Points. *Operations Research*, 12(4), 568–581.

2. Gendreau, M., Hertz, A., and Laporte, G. (1994). A Tabu Search Heuristic for the Vehicle Routing Problem. *Management Science*, 40(10), 1276–1290.

3. Glover, F. (1986). Future Paths for Integer Programming and Links to Artificial Intelligence. *Computers & Operations Research*, 13(5), 533–549.

4. Kirkpatrick, S., Gelatt, C.D., and Vecchi, M.P. (1983). Optimization by Simulated Annealing. *Science*, 220(4598), 671–680.

5. Lin, S. (1965). Computer Solutions of the Traveling Salesman Problem. *Bell System Technical Journal*, 44(10), 2245–2269.

6. Milburn, A.B., Kirac, E., and Hadianniasar, M. (2017). Growing Pains: A Case Study for Large-Scale Vehicle Routing. *INFORMS Transactions on Education*, 17(2), 81–84.

7. Or, I. (1976). Traveling Salesman-Type Combinatorial Problems and Their Relation to the Logistics of Regional Blood Banking. *PhD thesis*, Northwestern University.

8. Osman, I.H. (1993). Metastrategy Simulated Annealing and Tabu Search Algorithms for the Vehicle Routing Problems. *Annals of Operations Research*, 41, 421–451.

9. Ropke, S. and Pisinger, D. (2006). An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows. *Transportation Science*, 40(4), 455–472.

10. Shaw, P. (1998). Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems. *CP 1998*, Springer, 417–431.