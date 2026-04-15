# Methodology
### IE 7200 — Supply Chain Engineering | NHG Vehicle Routing Case Study

---

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
- $K$ — set of available vehicles (homogeneous fleet of vans)
- Depot located at ZIP code 01887 (Wilmington, MA)

**Parameters:**

| Symbol | Value | Description |
|---|---|---|
| $Q$ | 3,200 ft³ | Van capacity |
| $r$ | 0.03 min/ft³ | Unload rate |
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

**Capacity:** $\sum_{i \in \text{route}} \text{cube}_i \leq Q$

**Time windows:** Every stop $i$ must begin and complete service within $[08:00, 18:00]$. Service start time is $\max(\text{arrival}_i, 08:00)$; service completes at start time plus unload time.

**DOT HOS:** Total driving $\leq H_d$ and total on-duty time $\leq H_o$, where on-duty includes driving, unloading, and waiting time. Loading at the DC does not count toward on-duty hours.

### 1.4 Dispatch Logic

Vehicles are dispatched at the time required to arrive at the first stop exactly at 08:00:

$$t_{\text{dispatch}} = \max\left(0, 08:00 - \frac{d(\text{depot}, \text{first stop})}{v}\right)$$

The floor of zero prevents dispatch before midnight. On-duty time begins accumulating from dispatch.

### 1.5 Unload Time

$$t_{\text{unload}}(q) = \frac{\max(t_{\min},\ r \cdot q)}{60} \text{ hours}$$

---

## 2. Route Evaluation

All route evaluation is centralised in `vrp_solvers/base.py::evaluateRoute()`. This function is called millions of times during construction and improvement and is the computational bottleneck of every solver. It simulates the full route from dispatch to depot return and returns:

- Total miles, drive time, unload time, wait time, duty time
- Cube loaded
- Return time
- Three individual feasibility flags (capacity, window, DOT) and an overall feasibility flag

The function is intentionally simple and stateless — it takes a route list and returns a dict. No caching is applied; correctness is prioritised over speed at this scale.

---

## 3. Construction Heuristics

Construction heuristics build a complete feasible solution from scratch. They are fast, deterministic, and provide the starting point for all improvement methods.

### 3.1 Clarke-Wright Savings Algorithm

**Source:** Clarke and Wright (1964). Implemented in `vrp_solvers/clarkeWright.py`.

The algorithm starts with the most wasteful possible plan — a separate depot-to-store-to-depot round trip for every order — and iteratively merges pairs of routes using the savings criterion:

$$s(i, j) = d(\text{depot}, i) + d(\text{depot}, j) - d(i, j)$$

This measures how many miles are saved by visiting $i$ and $j$ on the same route rather than two separate ones. All $\binom{n}{2}$ pairs are computed and sorted descending. Merges are accepted in order when:

1. The saving is positive
2. Both orders are at valid endpoints of their current routes (tail-to-head adjacency)
3. The merged route passes all three feasibility constraints

Four merge orientations are attempted at each candidate pair to handle both forward and reversed adjacency, allowing the algorithm to construct routes in both directions from the depot.

**Complexity:** $O(n^2 \log n)$ for savings computation; merging is $O(n)$ per accepted pair.

**Route consolidation:** After CW construction, a post-processing phase attempts to eliminate the route with the smallest cube by relocating each of its stops to the cheapest feasible position in any remaining route. This is repeated until no further elimination is possible. Consolidation directly reduces vehicle count.

### 3.2 Nearest Neighbor Heuristic

**Source:** Classical VRP literature. Implemented in `vrp_solvers/nearestNeighbor.py`.

Starting from the depot, the algorithm repeatedly appends the closest unvisited store whose addition keeps the current route feasible. When no feasible extension exists, the current route is closed and a new one is opened from the depot.

The algorithm is $O(n^2)$ per day and significantly faster than CW, but produces lower-quality solutions because each greedy step forecloses options for future stops. Routes built by NN often end with a long backtrack to the depot because the algorithm commits to nearby stops first without regard for the return journey.

**Guard against infinite loop:** Orders whose cube alone exceeds van capacity are identified before the construction loop and excluded with a warning. A `madeProgress` flag detects iterations where no stop can be placed on any new route, breaking the outer loop rather than spinning indefinitely.

---

## 4. Local Search Improvement

Local search operators are applied after construction to improve route quality within a single vehicle's stops. They are deterministic, fast, and guarantee that the solution is locally optimal with respect to the moves they consider.

### 4.1 2-opt

**Source:** Lin (1965). Implemented in `vrp_solvers/base.py::twoOptRoute()`.

For every pair of edges $(i, i+1)$ and $(j, j+1)$ in a route, 2-opt considers reversing the segment between them. If the reversal is feasible and reduces total route miles, it is accepted. The process repeats until no improving 2-opt move exists — a condition called **2-optimal**.

2-opt eliminates all crossing edges in a route. Two edges that cross on a map can always be uncrossed by a reversal, and the uncrossed version is always shorter in Euclidean space. The algorithm runs in $O(n^2)$ per pass and typically requires several passes before convergence.

Applied to routes with 4 or more stops.

### 4.2 Or-opt

**Source:** Or (1976). Implemented in `vrp_solvers/base.py::orOptRoute()`.

Or-opt tries relocating **chains** of 1, 2, or 3 consecutive stops to every other position within the same route. A relocation is accepted if it is feasible and reduces total miles. Chain lengths are tried in order: single stops first, then pairs, then triples. The process repeats until no improving move exists.

Or-opt finds improvements that 2-opt cannot — particularly when a small group of stops is in the wrong position in the sequence. It is strictly more powerful than 2-opt for chain lengths 1-3 while remaining intra-route.

Applied to routes with 2 or more stops.

### 4.3 Combined local search pipeline

The standard improvement pipeline applied after construction is:

```
twoOptRoute()  →  orOptRoute()
```

Applied in `vrp_solvers/base.py::applyLocalSearch()`. Routes with fewer than 4 stops skip 2-opt (no reversals are possible) but still run Or-opt.

---

## 5. Metaheuristics

Metaheuristics guide local search to escape local optima through mechanisms that allow occasional acceptance of worse solutions, maintain memory of previously visited states, or explore large portions of the solution space simultaneously. All three metaheuristics implemented here seed from the CW + consolidation + 2-opt + Or-opt solution to ensure a fair starting point.

### 5.1 Tabu Search

**Source:** Glover (1986); applied to VRP by Gendreau, Hertz, and Laporte (1994). Implemented in `vrp_solvers/tabuSearch.py`.

#### Neighbourhood structure

Two inter-route move types are considered:

- **Relocate:** Move one stop from its current route to any position in any other route
- **Swap:** Exchange one stop from route $A$ with one stop from route $B$

At each iteration, all valid relocate and swap moves are generated (up to 200 sampled at random for efficiency), and the best non-tabu feasible move is accepted — even if it worsens the current solution. This forced acceptance of degrading moves is what drives the search out of local optima.

#### Tabu list

Recently performed moves are stored in a tabu list keyed by `(move_type, source_route_idx, source_position)`. A move remains tabu for `tabuTenure` iterations. The tabu list prevents cycling by forbidding the search from immediately reversing a move it just made.

#### Aspiration criterion

A tabu move is accepted if it produces a solution better than the global best found so far. This override ensures the tabu list never blocks a genuinely optimal move.

#### Parameters

| Parameter | Default | Effect |
|---|---|---|
| `maxIter` | 300 | Total iterations per day |
| `tabuTenure` | 15 | Iterations a move remains forbidden |

### 5.2 Simulated Annealing

**Source:** Kirkpatrick, Gelatt, and Vecchi (1983); applied to VRP by Osman (1993). Implemented in `vrp_solvers/simulatedAnnealing.py`.

#### Mechanism

At each iteration, a single random stop relocation is proposed. The acceptance decision follows the Metropolis criterion:

$$P(\text{accept}) = \begin{cases} 1 & \text{if } \Delta \leq 0 \\ \exp\left(-\Delta / T\right) & \text{if } \Delta > 0 \end{cases}$$

where $\Delta = \text{new miles} - \text{current miles}$ and $T$ is the current temperature. Improving moves are always accepted; worsening moves are accepted with probability that decays as temperature falls.

#### Cooling schedule

Temperature follows a geometric decay:

$$T_k = T_0 \cdot \alpha^k, \quad \alpha = \left(\frac{T_{\text{end}}}{T_0}\right)^{1/\text{maxIter}}$$

Early in the run, high temperature means the search explores broadly, accepting many worsening moves. As temperature approaches $T_{\text{end}}$, the algorithm converges toward pure local search. The best solution encountered at any point in the run is tracked and returned.

#### Parameters

| Parameter | Default | Effect |
|---|---|---|
| `maxIter` | 2000 | Total iterations per day |
| `tempStart` | 500.0 | Initial temperature |
| `tempEnd` | 1.0 | Final temperature |

### 5.3 Adaptive Large Neighborhood Search

**Source:** Shaw (1998) for LNS; Ropke and Pisinger (2006) for ALNS. Implemented in `vrp_solvers/alns.py`.

#### Core mechanism

At each iteration, ALNS performs a **destroy-repair** cycle:

1. A destroy operator removes a subset of stops from the current solution
2. A repair operator reinserts those stops to rebuild a complete solution

The destroyed portion is typically 20% of all stops (`removeFrac=0.2`). This large neighbourhood allows the algorithm to make structural changes that no sequence of single-stop moves could achieve.

#### Destroy operators

| Operator | Logic |
|---|---|
| Random removal | Remove $k$ stops chosen uniformly at random |
| Worst removal | Remove the $k$ stops contributing most to their route's cost |
| Shaw removal | Remove a seed stop and the $k-1$ geographically closest stops to encourage re-clustering |
| Route removal | Remove all stops from the smallest route, forcing complete redistribution |

#### Repair operators

| Operator | Logic |
|---|---|
| Greedy insertion | Insert each stop at the cheapest feasible position across all routes |
| Regret-2 insertion | Insert the stop whose cost difference between its best and second-best positions is largest first — reduces route lock-in |

#### Adaptive weight mechanism

Each operator pair maintains a weight updated after every `UPDATE_FREQ=50` iterations:

$$w_i \leftarrow \delta \cdot w_i + (1 - \delta) \cdot \frac{\text{score}_i}{\text{uses}_i}$$

where $\delta=0.8$ is the decay factor. Scores are assigned based on the quality of the improvement found:

| Event | Score |
|---|---|
| New global best | 9 |
| Better than current | 5 |
| Accepted (worse, via SA criterion) | 2 |
| Rejected | 0 |

Operator selection uses roulette-wheel sampling proportional to current weights. Weights are floored at 0.01 to prevent any operator from being permanently disabled.

#### Acceptance criterion

ALNS uses a simulated annealing acceptance criterion to occasionally accept worse solutions. Initial temperature is set to 5% of the seed solution's miles, decaying geometrically at rate 0.999 per iteration.

#### Parameters

| Parameter | Default | Effect |
|---|---|---|
| `maxIter` | 500 | Total iterations per day |
| `removeFrac` | 0.2 | Fraction of stops destroyed per iteration |

---

## 6. Overnight Route Extension

**Implemented in:** `vrp_solvers/overnightSolver.py`

### 6.1 Motivation

The base case treats each weekday independently, requiring every driver to return to the Wilmington depot on the same calendar day as their deliveries. This is operationally conservative. The overnight extension allows a driver to continue travelling toward the next day's first delivery stop after completing their day-1 deliveries, take the mandatory 10-hour DOT break en route, and resume on day 2.

This can reduce total miles when two adjacent-day routes serve geographically proximate stores — the driver avoids the return trip to Wilmington and a fresh outbound trip the following morning.

### 6.2 Delivery window constraint

All deliveries still occur within the 08:00–18:00 store operating window on their respective calendar day. The overnight break is a logistics event between deliveries, not during them. This constraint is enforced independently for each stop in `serveRouteSegment()`.

### 6.3 Break placement — latest break rule

After completing all day-1 deliveries, the driver travels toward the day-2 first stop until their HOS limit is reached, then takes the break. Two cases:

**Case 1 — Driver reaches day-2 stop before HOS runs out:**
The driver arrives at the next stop, takes the full 10-hour break on arrival, and begins day-2 service as soon as the break ends (or 08:00 if that is later).

**Case 2 — HOS runs out en route:**
The driver stops wherever the legal travel time runs out, takes the 10-hour break, then continues driving to the day-2 stop.

In both cases, day-2 HOS accumulators reset to zero when the break ends.

### 6.4 Overnight pairing algorithm

For every pair of adjacent-day routes $(r_1, r_2)$:

1. Evaluate the overnight route combining $r_1$ and $r_2$ using `evaluateOvernightRoute()`
2. Compare total miles to the sum of the two separate routes
3. If the overnight combination is feasible and saves miles, record it as a candidate

Candidates are sorted by savings descending and applied greedily — each route participates in at most one overnight pairing. This is implemented in `applyOvernightImprovements()`.

### 6.5 Scope

Overnight routes span exactly two consecutive calendar days. Routes cannot span three or more days. Only adjacent weekday pairs are considered: Mon-Tue, Tue-Wed, Wed-Thu, Thu-Fri.

---

## 7. Resource Analysis

**Implemented in:** `vrp_solvers/resourceAnalyser.py`

### 7.1 Truck requirements

One truck is required per simultaneous route. Since MAD loads trailers in advance and all routes dispatch concurrently from the same depot window, the minimum trucks needed on a given day equals the number of routes on that day. The weekly fleet requirement is the maximum across all five days.

### 7.2 Driver requirements — minimum path cover

The minimum driver count is solved as a **minimum path cover on a directed acyclic graph**:

- **Nodes:** Every (day, route_index) pair
- **Edges:** A directed edge from route $A$ on day $N$ to route $B$ on day $N+1$ exists when the driver finishing $A$ satisfies the 10-hour break before $B$'s dispatch time:

$$t_{\text{return}}(A) + 10 \leq t_{\text{dispatch}}(B) + 24$$

The $+24$ converts day-$N+1$ dispatch times (relative to midnight of that day) into absolute hours comparable to a day-$N$ return time.

By König's theorem, the minimum number of paths (drivers) needed to cover all nodes equals:

$$\text{min drivers} = \text{total routes} - \text{max bipartite matching}$$

The maximum bipartite matching is found using augmenting paths (Hopcroft-Karp style) implemented directly in Python — sufficient for the NHG scale of ~10 routes per day.

### 7.3 Overnight pairings

Overnight route pairs are pre-committed driver chains — one driver covers both the day-1 and day-2 portions of an overnight pairing. These are removed from the bipartite matching problem before solving, then added back as single pre-committed chains. Each overnight pairing contributes exactly one driver.

### 7.4 DOT 70-hour/8-day rule

The DOT 70-hour rule applies to a rolling 8-day window and is not modelled in this project. The dataset represents a single average week; inter-week driver continuity is outside the project scope. The 10-hour between-shift break is the binding HOS constraint within the single-week horizon.

---

## 8. Algorithm Comparison Matrix

| Configuration | Construction | Local Search | Metaheuristic | Overnight |
|---|---|---|---|---|
| `cw_only` | Clarke-Wright | — | — | — |
| `nn_only` | Nearest Neighbor | — | — | — |
| `cw_2opt_oropt` | Clarke-Wright | 2-opt + Or-opt | — | — |
| `nn_2opt_oropt` | Nearest Neighbor | 2-opt + Or-opt | — | — |
| `tabu_search` | CW | 2-opt + Or-opt (seed) | Tabu Search | — |
| `simulated_annealing` | CW | 2-opt + Or-opt (seed) | Simulated Annealing | — |
| `alns` | CW | 2-opt + Or-opt (seed) | ALNS | — |
| `cw_overnight` | CW | 2-opt + Or-opt | — | Yes |
| `alns_overnight` | CW | 2-opt + Or-opt (seed) | ALNS | Yes |

---

## 9. Software Architecture

### 9.1 Package structure

All solver logic lives in the `vrp_solvers/` package. Each algorithm is implemented as a class with a consistent interface:

```python
solver = ClarkeWrightSolver(useTwoOpt=True, useOrOpt=True)
routes = solver.solve(dayOrders)        # run algorithm, return routes
stats  = solver.getStats()              # miles, routes, feasibility, runtime
curve  = solver.getConvergence()        # convergence curve (None for construction)
```

The `OvernightSolver` wraps any base solver and takes full weekly orders rather than a single day:

```python
solver = OvernightSolver(ALNSSolver())
routesByDay, overnightRoutes, usedRoutes = solver.solve(orders)
```

The `ResourceAnalyser` takes a `routesByDay` dict and optional overnight pairings:

```python
analyser = ResourceAnalyser(routesByDay, overnightPairings=overnightRoutes)
analyser.analyse()
report = analyser.getReport()
```

### 9.2 Shared base module

`vrp_solvers/base.py` is the single source of truth for all constants, data loading, and shared computation. All solver classes import from it — no constant or utility function is defined more than once. The distance matrix is stored as a module-level global (`DIST_MATRIX`) populated by `loadInputs()`, making lookups O(1) without passing the matrix through function arguments.

### 9.3 Data flow

```
deliveries.xlsx  ──┐
distances.xlsx   ──┴──► VRP_DataAnalysis.py ──► data/
                                                   │
                         ┌─────────────────────────┘
                         ▼
                    vrp_solvers/base.py (loadInputs)
                         │
              ┌──────────┼──────────────────────┐
              ▼          ▼                      ▼
       VRP_BaseCase  VRP_BaseCase_Map   VRP_SolverComparison
                                               │
                                               ▼
                                        outputs/comparison/

VRP_OvernightRoutes.py ──► reads data/ directly, standalone
```

---

## 10. References

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
