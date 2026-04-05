import math
import pandas as pd

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

#load in data files
orders = pd.read_excel('deliveries.xlsx', sheet_name='OrderTable')
location = pd.read_excel('deliveries.xlsx', sheet_name='LocationTable')
distances = pd.read_excel('distances.xlsx', sheet_name='Sheet1')

#define constraints
van_capacity = 3200 #cubic feet
unload_rate = 0.03 #min/cubic foot
min_time = 30 #minutes per order
driving_speed = 40 #mph
window_open = 8 #start of delivery window
window_close = 18 #end of delivery window
break_time = 10 #break time is 10 hours
max_driving = 11 #11 hours max driving
max_duty = 14 #14 hours on duty, driving + waiting + loading
depot_zip = 1887 #depot zipcode

#clean data
orders = orders[orders["ORDERID"] != 0].copy()
orders["CUBE"] = pd.to_numeric(orders["CUBE"], errors="raise")
orders["FROMZIP"] = pd.to_numeric(orders["FROMZIP"], errors="raise")
orders["TOZIP"] = pd.to_numeric(orders["TOZIP"], errors="raise")
orders["ORDERID"] = pd.to_numeric(orders["ORDERID"], errors="raise")


distances = distances.rename(columns={
    "Unnamed: 0": "ZIP",
    "Unnamed: 1": "ZIPID"
})

distances = distances[distances["ZIP"] != "Zip"].copy()

distances["ZIP"] = pd.to_numeric(distances["ZIP"], errors="raise")
distances["ZIPID"] = pd.to_numeric(distances["ZIPID"], errors="raise")

# set row ZIP as the index
distances = distances.set_index("ZIP")

dist_matrix = distances.drop(columns=["ZIPID"]).copy()

order_zips = set(orders["TOZIP"].unique())
distance_zips = set(dist_matrix.index)

def get_distance(zip1, zip2):
    return dist_matrix.loc[zip1, zip2]

# unload time function
def get_unload_time(cube):
    return max(min_time, unload_rate * cube)/60   #hours

def to_clock(hours):
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h += 1
        m = 0
    return f"{h:02d}:{m:02d}"

def evaluate_route(route_list, verbose=False):
    current_zip = depot_zip
    total_miles = 0
    total_drive = 0
    total_unload = 0
    total_wait = 0
    total_cube = 0
    all_before_close = True

    first_zip = route_list[0]["TOZIP"]
    first_drive = get_distance(depot_zip, first_zip) / driving_speed
    current_time = window_open - first_drive

    if verbose:
        print("Dispatch:", to_clock(current_time))

    for stop in route_list:
        stop_zip = stop["TOZIP"]
        cube = stop["CUBE"]

        miles_leg = get_distance(current_zip, stop_zip)
        drive = miles_leg / driving_speed
        arrival = current_time + drive
        service_start = max(arrival, window_open)
        wait = max(0, window_open - arrival)
        unload = get_unload_time(cube)
        departure = service_start + unload

        before_close = service_start <= window_close
        all_before_close = all_before_close and before_close

        if verbose:
            print("Stop:", stop["ORDERID"])
            print(" Arrive:", to_clock(arrival))
            print(" Start service:", to_clock(service_start))
            print(" Depart:", to_clock(departure))
            print(" Before close:", before_close)

        total_miles += miles_leg
        total_drive += drive
        total_wait += wait
        total_unload += unload
        total_cube += cube

        current_time = departure
        current_zip = stop_zip

    miles_back = get_distance(current_zip, depot_zip)
    drive_back = miles_back / driving_speed
    return_time = current_time + drive_back

    total_miles += miles_back
    total_drive += drive_back
    total_duty = total_drive + total_unload + total_wait

    capacity_feasible = total_cube <= van_capacity
    dot_feasible = (total_drive <= max_driving) and (total_duty <= max_duty)
    overall_feasible = capacity_feasible and all_before_close and dot_feasible

    results = {
        "total_miles": int(total_miles),
        "total_drive": round(float(total_drive), 3),
        "total_unload": round(float(total_unload), 3),
        "total_wait": round(float(total_wait), 3),
        "total_duty": round(float(total_duty), 3),
        "total_cube": int(total_cube),
        "return_time": round(float(return_time), 3),
        "capacity_feasible": bool(capacity_feasible),
        "window_feasible": bool(all_before_close),
        "dot_feasible": bool(dot_feasible),
        "overall_feasible": bool(overall_feasible)
    }

    if verbose:
        print("\nReturn to depot:", to_clock(return_time))
        print("\nTotals")
        print(results)

    return results

def best_insertion(route_list, candidate_orders):
    best_order = None
    best_route = None
    best_results = None
    best_extra_miles = float("inf")

    base_results = evaluate_route(route_list, verbose=False)
    base_miles = base_results["total_miles"]

    for _, cand in candidate_orders.iterrows():
        for pos in range(len(route_list) + 1):
            trial_route = route_list[:pos] + [cand] + route_list[pos:]
            results = evaluate_route(trial_route, verbose=False)

            if results["overall_feasible"]:
                extra_miles = results["total_miles"] - base_miles

                if extra_miles < best_extra_miles:
                    best_order = cand
                    best_route = trial_route
                    best_results = results
                    best_extra_miles = extra_miles

    return best_order, best_route, best_results

def build_route_from_seed(day_orders, seed):
    route_list = [seed]
    remaining = day_orders[day_orders["ORDERID"] != seed["ORDERID"]].copy()

    while True:
        best_order, best_route, best_results = best_insertion(route_list, remaining)
        if best_order is None:
            break

        route_list = best_route
        remaining = remaining[remaining["ORDERID"] != best_order["ORDERID"]].copy()

    return route_list, remaining

def build_one_route(day_orders):
    candidate_seeds = []

    # first remaining
    candidate_seeds.append(day_orders.iloc[0])

    # farthest from depot
    farthest_idx = day_orders["TOZIP"].apply(lambda z: get_distance(depot_zip, z)).idxmax()
    candidate_seeds.append(day_orders.loc[farthest_idx])

    # largest cube
    largest_idx = day_orders["CUBE"].idxmax()
    candidate_seeds.append(day_orders.loc[largest_idx])

    # remove duplicates
    unique_seeds = []
    seen_ids = set()
    for seed in candidate_seeds:
        oid = int(seed["ORDERID"])
        if oid not in seen_ids:
            unique_seeds.append(seed)
            seen_ids.add(oid)

    best_route = None
    best_remaining = None
    best_score = float("inf")

    for seed in unique_seeds:
        route_list, remaining = build_route_from_seed(day_orders, seed)
        results = evaluate_route(route_list, verbose=False)

        # simple score: lower miles is better
        score = results["total_miles"]

        if score < best_score:
            best_score = score
            best_route = route_list
            best_remaining = remaining

    return best_route, best_remaining

def build_all_routes_for_day(day_orders):
    remaining = day_orders.copy()
    all_routes = []

    while len(remaining) > 0:
        route_list, remaining = build_one_route(remaining)
        all_routes.append(route_list)

    return all_routes

def best_relocation(stop, routes, skip_route_idx):
    best_target_idx = None
    best_new_route = None
    best_extra_miles = float("inf")

    stop_df = pd.DataFrame([stop])

    for j, route in enumerate(routes):
        if j == skip_route_idx:
            continue

        base_results = evaluate_route(route, verbose=False)
        base_miles = base_results["total_miles"]

        for pos in range(len(route) + 1):
            trial_route = route[:pos] + [stop] + route[pos:]
            results = evaluate_route(trial_route, verbose=False)

            if results["overall_feasible"]:
                extra_miles = results["total_miles"] - base_miles

                if extra_miles < best_extra_miles:
                    best_target_idx = j
                    best_new_route = trial_route
                    best_extra_miles = extra_miles

    return best_target_idx, best_new_route, best_extra_miles

def try_eliminate_one_route(all_routes):
    route_infos = []
    for i, route in enumerate(all_routes):
        results = evaluate_route(route, verbose=False)
        route_infos.append((i, results["total_cube"], results["total_miles"]))

    # try small/weak routes first
    route_infos.sort(key=lambda x: (x[1], x[2]))

    for remove_idx, _, _ in route_infos:
        route_to_remove = all_routes[remove_idx]
        new_routes = all_routes.copy()
        success = True

        for stop in route_to_remove:
            target_idx, new_target_route, extra_miles = best_relocation(stop, new_routes, remove_idx)

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
    best_results = evaluate_route(best_route, verbose=False)
    improved = True

    while improved:
        improved = False

        # need at least 4 stops to make 2-opt meaningful
        for i in range(len(best_route) - 1):
            for j in range(i + 2, len(best_route)):
                trial_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                trial_results = evaluate_route(trial_route, verbose=False)

                if trial_results["overall_feasible"] and trial_results["total_miles"] < best_results["total_miles"]:
                    best_route = trial_route
                    best_results = trial_results
                    improved = True
                    break

            if improved:
                break

    return best_route

def improve_routes_by_2opt(all_routes):
    improved_routes = []

    for route in all_routes:
        if len(route) >= 4:
            improved_routes.append(two_opt_route(route))
        else:
            improved_routes.append(route)

    return improved_routes

days = ["Mon", "Tue", "Wed", "Thu", "Fri"]

weekly_total_miles = 0
weekly_total_routes = 0

for day in days:
    day_orders = orders[orders["DayOfWeek"] == day].copy()
    all_routes = build_all_routes_for_day(day_orders)
    all_routes = consolidate_routes(all_routes)
    all_routes = improve_routes_by_2opt(all_routes)

    print(f"\n===== {day} =====")

    day_total_miles = 0

    for i, route in enumerate(all_routes, start=1):
        results = evaluate_route(route, verbose=False)
        route_ids = [int(stop["ORDERID"]) for stop in route]

        print(f"Route {i}: {route_ids}")
        print(results)
        print()

        day_total_miles += results["total_miles"]

    print(f"{day} route count:", len(all_routes))
    print(f"{day} total miles:", day_total_miles)

    weekly_total_miles += day_total_miles
    weekly_total_routes += len(all_routes)

print("\n===== WEEKLY SUMMARY =====")
print("Total weekly routes:", weekly_total_routes)
print("Total weekly miles:", weekly_total_miles)