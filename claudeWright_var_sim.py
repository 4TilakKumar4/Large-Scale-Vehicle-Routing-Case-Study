import pandas as pd
import numpy as np
import warnings
from itertools import combinations
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# =========================
# Constants
# =========================
VAN_CAPACITY  = 3200
UNLOAD_RATE   = 0.03
MIN_TIME      = 30
DRIVING_SPEED = 40
WINDOW_OPEN   = 8
WINDOW_CLOSE  = 18
DEPOT_ZIP     = 1887
DAYS          = ["Mon", "Tue", "Wed", "Thu", "Fri"]

N_RUNS        = 50
NOISE_STD     = 5.0   # miles of Gaussian noise added to each savings score per run


# =========================
# Data loading
# =========================
def load_inputs():
    orders = pd.read_excel("deliveries.xlsx", sheet_name="OrderTable")
    distances = pd.read_excel("distances.xlsx", sheet_name="Sheet1")

    orders = orders[orders["ORDERID"] != 0].copy()
    for col in ["CUBE", "FROMZIP", "TOZIP", "ORDERID"]:
        orders[col] = pd.to_numeric(orders[col], errors="raise")

    distances = distances.rename(columns={"Unnamed: 0": "ZIP", "Unnamed: 1": "ZIPID"})
    distances = distances[distances["ZIP"] != "Zip"].copy()
    distances["ZIP"]   = pd.to_numeric(distances["ZIP"],   errors="raise")
    distances["ZIPID"] = pd.to_numeric(distances["ZIPID"], errors="raise")
    distances = distances.set_index("ZIP")

    dist_matrix = distances.drop(columns=["ZIPID"]).copy()
    dist_matrix.columns = pd.to_numeric(dist_matrix.columns, errors="coerce")

    return orders, dist_matrix


# =========================
# Helpers
# =========================
def get_distance(zip1, zip2):
    return DIST_MATRIX.loc[zip1, zip2]


def get_unload_time(cube):
    return max(MIN_TIME, UNLOAD_RATE * cube) / 60.0


def route_ids(route):
    return [int(s["ORDERID"]) for s in route]


# =========================
# Route evaluator
# =========================
def evaluate_route(route_list):
    first_zip    = route_list[0]["TOZIP"]
    first_drive  = get_distance(DEPOT_ZIP, first_zip) / DRIVING_SPEED
    current_time = max(0.0, WINDOW_OPEN - first_drive)
    current_zip  = DEPOT_ZIP

    total_miles = total_drive = total_unload = total_wait = total_cube = 0
    window_feasible = True

    for stop in route_list:
        stop_zip      = stop["TOZIP"]
        cube          = stop["CUBE"]
        miles_leg     = get_distance(current_zip, stop_zip)
        drive         = miles_leg / DRIVING_SPEED
        arrival       = current_time + drive
        service_start = max(arrival, WINDOW_OPEN)
        wait          = max(0.0, WINDOW_OPEN - arrival)
        unload        = get_unload_time(cube)
        departure     = service_start + unload

        tod          = service_start % 24
        eos          = departure % 24
        before_close = (WINDOW_OPEN <= tod <= WINDOW_CLOSE) and (eos <= WINDOW_CLOSE)
        window_feasible = window_feasible and before_close

        total_miles  += miles_leg
        total_drive  += drive
        total_wait   += wait
        total_unload += unload
        total_cube   += cube
        current_time  = departure
        current_zip   = stop_zip

    miles_back   = get_distance(current_zip, DEPOT_ZIP)
    drive_back   = miles_back / DRIVING_SPEED
    total_miles += miles_back
    total_drive += drive_back
    total_duty   = total_drive + total_unload + total_wait

    return {
        "total_miles":       int(total_miles),
        "total_drive":       round(float(total_drive), 3),
        "total_unload":      round(float(total_unload), 3),
        "total_wait":        round(float(total_wait), 3),
        "total_duty":        round(float(total_duty), 3),
        "total_cube":        int(total_cube),
        "capacity_feasible": bool(total_cube <= VAN_CAPACITY),
        "window_feasible":   bool(window_feasible),
        "overall_feasible":  bool((total_cube <= VAN_CAPACITY) and window_feasible),
    }


# =========================
# Clarke-Wright (with noise)
# =========================
def compute_savings(orders_df, rng):
    order_list = orders_df.to_dict("records")
    savings = []
    for a, b in combinations(order_list, 2):
        zip_a, zip_b = a["TOZIP"], b["TOZIP"]
        s = (
            get_distance(DEPOT_ZIP, zip_a)
            + get_distance(DEPOT_ZIP, zip_b)
            - get_distance(zip_a, zip_b)
            + rng.normal(0, NOISE_STD)
        )
        savings.append((s, int(a["ORDERID"]), int(b["ORDERID"])))
    savings.sort(key=lambda x: x[0], reverse=True)
    return savings


def clarke_wright(day_orders, rng):
    order_records = {int(row["ORDERID"]): row for _, row in day_orders.iterrows()}

    routes   = {oid: [rec] for oid, rec in order_records.items()}
    route_of = {oid: oid   for oid in order_records}
    head_of  = {oid: oid   for oid in order_records}
    tail_of  = {oid: oid   for oid in order_records}

    for s, oid_i, oid_j in compute_savings(day_orders, rng):
        if s <= 0:
            break
        if oid_i not in route_of or oid_j not in route_of:
            continue
        rid_i, rid_j = route_of[oid_i], route_of[oid_j]
        if rid_i == rid_j:
            continue

        route_i, route_j = routes[rid_i], routes[rid_j]
        candidates = []
        if tail_of[rid_i] == oid_i and head_of[rid_j] == oid_j:
            candidates.append((route_i + route_j, rid_i, rid_j))
        if tail_of[rid_j] == oid_j and head_of[rid_i] == oid_i:
            candidates.append((route_j + route_i, rid_j, rid_i))
        if tail_of[rid_i] == oid_i and tail_of[rid_j] == oid_j:
            candidates.append((route_i + route_j[::-1], rid_i, rid_j))
        if head_of[rid_i] == oid_i and head_of[rid_j] == oid_j:
            candidates.append((route_i[::-1] + route_j, rid_i, rid_j))

        for merged_route, keep_rid, drop_rid in candidates:
            if evaluate_route(merged_route)["overall_feasible"]:
                routes[keep_rid] = merged_route
                del routes[drop_rid]
                head_of[keep_rid] = int(merged_route[0]["ORDERID"])
                tail_of[keep_rid] = int(merged_route[-1]["ORDERID"])
                for stop in merged_route:
                    route_of[int(stop["ORDERID"])] = keep_rid
                del head_of[drop_rid], tail_of[drop_rid]
                break

    return list(routes.values())


# =========================
# Improvement heuristics
# =========================
def best_relocation(stop, routes, skip_idx):
    best_idx, best_route, best_extra = None, None, float("inf")
    for j, route in enumerate(routes):
        if j == skip_idx:
            continue
        base = evaluate_route(route)["total_miles"]
        for pos in range(len(route) + 1):
            trial = route[:pos] + [stop] + route[pos:]
            r = evaluate_route(trial)
            if r["overall_feasible"] and r["total_miles"] - base < best_extra:
                best_idx, best_route, best_extra = j, trial, r["total_miles"] - base
    return best_idx, best_route


def try_eliminate_one_route(all_routes):
    infos = sorted(
        [(i, evaluate_route(r)) for i, r in enumerate(all_routes)],
        key=lambda x: (x[1]["total_cube"], x[1]["total_miles"])
    )
    for remove_idx, _ in infos:
        new_routes = all_routes.copy()
        success = True
        for stop in all_routes[remove_idx]:
            tgt, new_r = best_relocation(stop, new_routes, remove_idx)
            if tgt is None:
                success = False; break
            new_routes[tgt] = new_r
        if success:
            new_routes.pop(remove_idx)
            return new_routes, True
    return all_routes, False


def consolidate_routes(all_routes):
    improved = True
    while improved:
        all_routes, improved = try_eliminate_one_route(all_routes)
    return all_routes


def two_opt_route(route):
    best, best_miles = route[:], evaluate_route(route)["total_miles"]
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                trial = best[:i] + best[i:j+1][::-1] + best[j+1:]
                r = evaluate_route(trial)
                if r["overall_feasible"] and r["total_miles"] < best_miles:
                    best, best_miles, improved = trial, r["total_miles"], True
                    break
            if improved:
                break
    return best


def or_opt_route(route):
    best, best_miles = route[:], evaluate_route(route)["total_miles"]
    improved = True
    while improved:
        improved = False
        for chain_len in [1, 2, 3]:
            for i in range(len(best) - chain_len + 1):
                chain     = best[i:i + chain_len]
                remainder = best[:i] + best[i + chain_len:]
                for j in range(len(remainder) + 1):
                    if j == i:
                        continue
                    trial = remainder[:j] + chain + remainder[j:]
                    r = evaluate_route(trial)
                    if r["overall_feasible"] and r["total_miles"] < best_miles:
                        best, best_miles, improved = trial, r["total_miles"], True
                        break
                if improved: break
            if improved: break
    return best


def improve_routes(all_routes):
    out = []
    for route in all_routes:
        if len(route) >= 4:
            route = two_opt_route(route)
        if len(route) >= 2:
            route = or_opt_route(route)
        out.append(route)
    return out


# =========================
# Single simulation run
# =========================
def run_once(seed):
    rng = np.random.default_rng(seed)
    routes_by_day = {}

    for day in DAYS:
        day_orders = ORDERS[ORDERS["DayOfWeek"] == day].copy()
        routes = clarke_wright(day_orders, rng)
        routes = consolidate_routes(routes)
        routes = improve_routes(routes)
        routes_by_day[day] = routes

    rows = []
    weekly_miles = 0

    for day in DAYS:
        for route_num, route in enumerate(routes_by_day[day], start=1):
            r = evaluate_route(route)
            weekly_miles += r["total_miles"]
            rows.append({
                "run":       seed,
                "day":       day,
                "route_num": route_num,
                "orders":    str(route_ids(route)),
                "n_stops":   len(route),
                "miles":     r["total_miles"],
                "cube":      r["total_cube"],
                "duty_hrs":  r["total_duty"],
                "feasible":  r["overall_feasible"],
            })

    return rows, weekly_miles


# =========================
# Main
# =========================
def main():
    all_rows     = []
    weekly_miles = []

    for run_num in range(N_RUNS):
        print(f"Run {run_num + 1}/{N_RUNS}...", end="\r", flush=True)
        rows, wm = run_once(seed=run_num)
        all_rows.extend(rows)
        weekly_miles.append(wm)

    print(f"\nAll {N_RUNS} runs complete.")

    # --- Route CSV ---
    routes_df = pd.DataFrame(all_rows)
    routes_df.to_csv("simulation_routes.csv", index=False)
    print("Routes saved  -> simulation_routes.csv")

    # --- Summary CSV ---
    wm_array        = np.array(weekly_miles)
    mean_wm         = wm_array.mean()
    std_wm          = wm_array.std(ddof=1)
    se_wm           = stats.sem(wm_array)
    ci_low, ci_high = stats.t.interval(0.95, df=N_RUNS - 1, loc=mean_wm, scale=se_wm)

    summary_df = pd.DataFrame({
        "run":          list(range(N_RUNS)),
        "weekly_miles": weekly_miles,
    })
    summary_df["mean"]    = mean_wm
    summary_df["ci_low"]  = round(ci_low,  1)
    summary_df["ci_high"] = round(ci_high, 1)
    summary_df.to_csv("simulation_summary.csv", index=False)
    print("Summary saved -> simulation_summary.csv")

    # --- Console report ---
    print("\n===== WEEKLY MILES ACROSS 50 RUNS =====")
    print(f"  Min:      {wm_array.min()} miles  (Run {int(wm_array.argmin())})")
    print(f"  Max:      {wm_array.max()} miles  (Run {int(wm_array.argmax())})")
    print(f"  Mean:     {mean_wm:.1f} miles")
    print(f"  Std Dev:  {std_wm:.1f} miles")
    print(f"  95% CI:   [{ci_low:.1f},  {ci_high:.1f}]")


ORDERS, DIST_MATRIX = load_inputs()

if __name__ == "__main__":
    main()