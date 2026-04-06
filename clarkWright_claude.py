import pandas as pd
import warnings
from itertools import combinations

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# =========================
# Constants
# =========================
VAN_CAPACITY = 3200
UNLOAD_RATE = 0.03          # min / cubic foot
MIN_TIME = 30               # minutes
DRIVING_SPEED = 40          # mph
WINDOW_OPEN = 8             # 8:00
WINDOW_CLOSE = 18           # 18:00
DEPOT_ZIP = 1887

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# =========================
# Data loading / cleaning
# =========================
def load_inputs():
    orders = pd.read_excel("deliveries.xlsx", sheet_name="OrderTable")
    distances = pd.read_excel("distances.xlsx", sheet_name="Sheet1")

    orders = orders[orders["ORDERID"] != 0].copy()
    orders["CUBE"] = pd.to_numeric(orders["CUBE"], errors="raise")
    orders["FROMZIP"] = pd.to_numeric(orders["FROMZIP"], errors="raise")
    orders["TOZIP"] = pd.to_numeric(orders["TOZIP"], errors="raise")
    orders["ORDERID"] = pd.to_numeric(orders["ORDERID"], errors="raise")

    distances = distances.rename(columns={"Unnamed: 0": "ZIP", "Unnamed: 1": "ZIPID"})
    distances = distances[distances["ZIP"] != "Zip"].copy()
    distances["ZIP"] = pd.to_numeric(distances["ZIP"], errors="raise")
    distances["ZIPID"] = pd.to_numeric(distances["ZIPID"], errors="raise")
    distances = distances.set_index("ZIP")

    dist_matrix = distances.drop(columns=["ZIPID"]).copy()
    dist_matrix.columns = pd.to_numeric(dist_matrix.columns, errors="coerce")

    return orders, dist_matrix


# =========================
# Small helpers
# =========================
def get_distance(zip1, zip2):
    return DIST_MATRIX.loc[zip1, zip2]


def get_unload_time(cube):
    return max(MIN_TIME, UNLOAD_RATE * cube) / 60.0


def to_clock(hours):
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h += 1
        m = 0
    return f"{h:02d}:{m:02d}"


def route_ids(route):
    return [int(stop["ORDERID"]) for stop in route]


# =========================
# Route evaluator
# =========================
def evaluate_route(route_list, verbose=False):
    first_zip = route_list[0]["TOZIP"]
    first_drive = get_distance(DEPOT_ZIP, first_zip) / DRIVING_SPEED
    dispatch_time = max(0.0, WINDOW_OPEN - first_drive)

    current_zip = DEPOT_ZIP
    current_time = dispatch_time

    total_miles = 0
    total_drive = 0
    total_unload = 0
    total_wait = 0
    total_cube = 0
    window_feasible = True

    if verbose:
        print("Dispatch:", to_clock(dispatch_time))

    for stop in route_list:
        stop_zip = stop["TOZIP"]
        cube = stop["CUBE"]

        miles_leg = get_distance(current_zip, stop_zip)
        drive = miles_leg / DRIVING_SPEED
        arrival = current_time + drive
        service_start = max(arrival, WINDOW_OPEN)
        wait = max(0.0, WINDOW_OPEN - arrival)
        unload = get_unload_time(cube)
        departure = service_start + unload

        time_of_day = service_start % 24
        end_of_service = departure % 24
        before_close = (WINDOW_OPEN <= time_of_day <= WINDOW_CLOSE) and (end_of_service <= WINDOW_CLOSE)
        window_feasible = window_feasible and before_close

        if verbose:
            print(f"  Stop {int(stop['ORDERID'])}: arrive {to_clock(arrival)} | "
                  f"start {to_clock(service_start)} | depart {to_clock(departure)} | "
                  f"ok={before_close}")

        total_miles += miles_leg
        total_drive += drive
        total_wait += wait
        total_unload += unload
        total_cube += cube

        current_time = departure
        current_zip = stop_zip

    miles_back = get_distance(current_zip, DEPOT_ZIP)
    drive_back = miles_back / DRIVING_SPEED
    return_time = current_time + drive_back

    total_miles += miles_back
    total_drive += drive_back
    total_duty = total_drive + total_unload + total_wait

    capacity_feasible = total_cube <= VAN_CAPACITY
    overall_feasible = capacity_feasible and window_feasible

    results = {
        "total_miles": int(total_miles),
        "total_drive": round(float(total_drive), 3),
        "total_unload": round(float(total_unload), 3),
        "total_wait": round(float(total_wait), 3),
        "total_duty": round(float(total_duty), 3),
        "total_cube": int(total_cube),
        "return_time": round(float(return_time), 3),
        "capacity_feasible": bool(capacity_feasible),
        "window_feasible": bool(window_feasible),
        "overall_feasible": bool(overall_feasible),
    }

    if verbose:
        print("Return to depot:", to_clock(return_time))
        print("Totals:", results)

    return results


# =========================
# Clarke-Wright construction
# =========================
def compute_savings(orders_df):
    order_list = orders_df.to_dict("records")
    savings = []

    for a, b in combinations(order_list, 2):
        zip_a = a["TOZIP"]
        zip_b = b["TOZIP"]
        s = (
            get_distance(DEPOT_ZIP, zip_a)
            + get_distance(DEPOT_ZIP, zip_b)
            - get_distance(zip_a, zip_b)
        )
        savings.append((s, int(a["ORDERID"]), int(b["ORDERID"])))

    savings.sort(key=lambda x: x[0], reverse=True)
    return savings


def clarke_wright(day_orders):
    order_records = {int(row["ORDERID"]): row for _, row in day_orders.iterrows()}

    routes   = {oid: [rec] for oid, rec in order_records.items()}
    route_of = {oid: oid   for oid in order_records}
    head_of  = {oid: oid   for oid in order_records}
    tail_of  = {oid: oid   for oid in order_records}

    for s, oid_i, oid_j in compute_savings(day_orders):
        if s <= 0:
            break

        if oid_i not in route_of or oid_j not in route_of:
            continue

        rid_i = route_of[oid_i]
        rid_j = route_of[oid_j]

        if rid_i == rid_j:
            continue

        route_i = routes[rid_i]
        route_j = routes[rid_j]

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
            if evaluate_route(merged_route, verbose=False)["overall_feasible"]:
                routes[keep_rid] = merged_route
                del routes[drop_rid]

                head_of[keep_rid] = int(merged_route[0]["ORDERID"])
                tail_of[keep_rid] = int(merged_route[-1]["ORDERID"])

                for stop in merged_route:
                    route_of[int(stop["ORDERID"])] = keep_rid

                del head_of[drop_rid]
                del tail_of[drop_rid]
                break

    return list(routes.values())


# =========================
# Improvement heuristics
# =========================
def best_relocation(stop, routes, skip_route_idx):
    best_target_idx = None
    best_new_route = None
    best_extra_miles = float("inf")

    for j, route in enumerate(routes):
        if j == skip_route_idx:
            continue

        base_miles = evaluate_route(route, verbose=False)["total_miles"]

        for pos in range(len(route) + 1):
            trial_route = route[:pos] + [stop] + route[pos:]
            results = evaluate_route(trial_route, verbose=False)

            if results["overall_feasible"]:
                extra_miles = results["total_miles"] - base_miles
                if extra_miles < best_extra_miles:
                    best_target_idx = j
                    best_new_route = trial_route
                    best_extra_miles = extra_miles

    return best_target_idx, best_new_route


def try_eliminate_one_route(all_routes):
    route_infos = sorted(
        [(i, evaluate_route(r, verbose=False)) for i, r in enumerate(all_routes)],
        key=lambda x: (x[1]["total_cube"], x[1]["total_miles"])
    )

    for remove_idx, _ in route_infos:
        route_to_remove = all_routes[remove_idx]
        new_routes = all_routes.copy()
        success = True

        for stop in route_to_remove:
            target_idx, new_target_route = best_relocation(stop, new_routes, remove_idx)
            if target_idx is None:
                success = False
                break
            new_routes[target_idx] = new_target_route

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
    best_route = route[:]
    best_miles = evaluate_route(best_route, verbose=False)["total_miles"]
    improved = True

    while improved:
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 2, len(best_route)):
                trial = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                result = evaluate_route(trial, verbose=False)

                if result["overall_feasible"] and result["total_miles"] < best_miles:
                    best_route = trial
                    best_miles = result["total_miles"]
                    improved = True
                    break
            if improved:
                break

    return best_route


def or_opt_route(route):
    best_route = route[:]
    best_miles = evaluate_route(best_route, verbose=False)["total_miles"]
    improved = True

    while improved:
        improved = False
        for chain_len in [1, 2, 3]:
            for i in range(len(best_route) - chain_len + 1):
                chain = best_route[i:i + chain_len]
                remainder = best_route[:i] + best_route[i + chain_len:]

                for j in range(len(remainder) + 1):
                    if j == i:
                        continue
                    trial = remainder[:j] + chain + remainder[j:]
                    result = evaluate_route(trial, verbose=False)

                    if result["overall_feasible"] and result["total_miles"] < best_miles:
                        best_route = trial
                        best_miles = result["total_miles"]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return best_route


def improve_routes(all_routes):
    improved = []
    for route in all_routes:
        if len(route) >= 4:
            route = two_opt_route(route)
        if len(route) >= 2:
            route = or_opt_route(route)
        improved.append(route)
    return improved


# =========================
# Reporting
# =========================
def print_day_report(day, routes):
    print(f"\n===== {day} =====")
    day_total_miles = 0
    day_total_orders = 0

    for i, route in enumerate(routes, start=1):
        r = evaluate_route(route, verbose=False)
        order_count = len(route)
        print(
            f"  Route {i}: {route_ids(route)} | "
            f"orders={order_count} | "
            f"miles={r['total_miles']} | cube={r['total_cube']} | "
            f"duty={r['total_duty']} | feasible={r['overall_feasible']}"
        )
        day_total_miles += r["total_miles"]
        day_total_orders += order_count

    print(f"  {day} routes: {len(routes)} | orders: {day_total_orders} | total miles: {day_total_miles}")
    return day_total_miles, day_total_orders


# =========================
# Main
# =========================
def main():
    routes_by_day = {}

    for day in DAYS:
        print(f"Building routes for {day}...")
        day_orders = ORDERS[ORDERS["DayOfWeek"] == day].copy()
        routes = clarke_wright(day_orders)
        routes = consolidate_routes(routes)
        routes = improve_routes(routes)
        routes_by_day[day] = routes

    weekly_total_miles = 0
    weekly_total_routes = 0
    weekly_total_orders = 0

    for day in DAYS:
        day_miles, day_orders = print_day_report(day, routes_by_day[day])
        weekly_total_miles += day_miles
        weekly_total_routes += len(routes_by_day[day])
        weekly_total_orders += day_orders

    print("\n===== WEEKLY SUMMARY =====")
    print("Total routes:", weekly_total_routes)
    print("Total orders fulfilled:", weekly_total_orders)
    print("Total miles:", weekly_total_miles)


ORDERS, DIST_MATRIX = load_inputs()

if __name__ == "__main__":
    main()