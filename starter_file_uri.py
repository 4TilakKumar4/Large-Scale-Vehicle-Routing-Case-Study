import pandas as pd
import warnings

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
BREAK_TIME = 10             # hours
MAX_DRIVING = 11            # hours
MAX_DUTY = 14               # hours
DEPOT_ZIP = 1887

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
ADJACENT_DAY_PAIRS = [("Mon", "Tue"), ("Tue", "Wed"), ("Wed", "Thu"), ("Thu", "Fri")]


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
# Route evaluators
# =========================
def serve_route_segment(route_list, start_zip, start_time, drive_used=0, duty_used=0, verbose=False):
    current_zip = start_zip
    current_time = start_time

    total_miles = 0
    total_drive = drive_used
    total_unload = 0
    total_wait = 0
    total_cube = 0
    window_feasible = True

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
        before_close = WINDOW_OPEN <= time_of_day <= WINDOW_CLOSE
        window_feasible = window_feasible and before_close

        if verbose:
            print("Stop:", int(stop["ORDERID"]))
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

    total_duty = duty_used + total_unload + total_wait + (total_drive - drive_used)

    return {
        "end_zip": current_zip,
        "end_time": current_time,
        "total_miles": total_miles,
        "total_drive": total_drive,
        "total_unload": total_unload,
        "total_wait": total_wait,
        "total_duty": total_duty,
        "total_cube": total_cube,
        "window_feasible": window_feasible,
    }


def evaluate_route(route_list, verbose=False):
    first_zip = route_list[0]["TOZIP"]
    first_drive = get_distance(DEPOT_ZIP, first_zip) / DRIVING_SPEED
    dispatch_time = WINDOW_OPEN - first_drive

    if verbose:
        print("Dispatch:", to_clock(dispatch_time))

    seg = serve_route_segment(
        route_list,
        start_zip=DEPOT_ZIP,
        start_time=dispatch_time,
        drive_used=0,
        duty_used=0,
        verbose=verbose,
    )

    miles_back = get_distance(seg["end_zip"], DEPOT_ZIP)
    drive_back = miles_back / DRIVING_SPEED
    return_time = seg["end_time"] + drive_back

    total_miles = seg["total_miles"] + miles_back
    total_drive = seg["total_drive"] + drive_back
    total_unload = seg["total_unload"]
    total_wait = seg["total_wait"]
    total_duty = total_drive + total_unload + total_wait
    total_cube = seg["total_cube"]

    capacity_feasible = total_cube <= VAN_CAPACITY
    dot_feasible = total_drive <= MAX_DRIVING and total_duty <= MAX_DUTY
    overall_feasible = capacity_feasible and seg["window_feasible"] and dot_feasible

    results = {
        "total_miles": int(total_miles),
        "total_drive": round(float(total_drive), 3),
        "total_unload": round(float(total_unload), 3),
        "total_wait": round(float(total_wait), 3),
        "total_duty": round(float(total_duty), 3),
        "total_cube": int(total_cube),
        "return_time": round(float(return_time), 3),
        "capacity_feasible": bool(capacity_feasible),
        "window_feasible": bool(seg["window_feasible"]),
        "dot_feasible": bool(dot_feasible),
        "overall_feasible": bool(overall_feasible),
    }

    if verbose:
        print("\nReturn to depot:", to_clock(return_time))
        print("\nTotals")
        print(results)

    return results


def latest_break_transition(last_zip, first_next_zip, finish_time, drive_used, duty_used):
    remaining_drive = MAX_DRIVING - drive_used
    remaining_duty = MAX_DUTY - duty_used
    legal_travel_hours = min(remaining_drive, remaining_duty)

    if legal_travel_hours < 0:
        return {"feasible": False, "reason": "day 1 already exceeds drive/duty limit"}

    travel_needed_hours = get_distance(last_zip, first_next_zip) / DRIVING_SPEED
    day2_open_time = 24 + WINDOW_OPEN

    if travel_needed_hours <= legal_travel_hours:
        arrival_B_day1 = finish_time + travel_needed_hours
        break_start_time = finish_time + legal_travel_hours
        wait_at_B_day1 = break_start_time - arrival_B_day1
        break_end_time = break_start_time + BREAK_TIME

        arrival_B_day2 = break_end_time
        day2_wait_at_B = max(0.0, day2_open_time - arrival_B_day2)
        service_start_B = arrival_B_day2 + day2_wait_at_B

        return {
            "feasible": True,
            "case": "reach_B_before_break",
            "day1_added_miles": get_distance(last_zip, first_next_zip),
            "day1_added_drive": travel_needed_hours,
            "day1_added_wait": wait_at_B_day1,
            "day1_final_drive": drive_used + travel_needed_hours,
            "day1_final_duty": duty_used + travel_needed_hours + wait_at_B_day1,
            "break_start_time": break_start_time,
            "break_end_time": break_end_time,
            "day2_arrival_B": arrival_B_day2,
            "service_start_B": service_start_B,
            "day2_drive_used_before_service": 0.0,
            "day2_duty_used_before_service": day2_wait_at_B,
            "day2_added_miles": 0.0,
            "day2_added_wait": day2_wait_at_B,
        }

    break_start_time = finish_time + legal_travel_hours
    break_end_time = break_start_time + BREAK_TIME

    day1_added_drive = legal_travel_hours
    day1_added_miles = legal_travel_hours * DRIVING_SPEED

    remaining_hours_to_B_after_break = travel_needed_hours - legal_travel_hours
    arrival_B_day2 = break_end_time + remaining_hours_to_B_after_break
    day2_wait_at_B = max(0.0, day2_open_time - arrival_B_day2)
    service_start_B = arrival_B_day2 + day2_wait_at_B

    return {
        "feasible": True,
        "case": "break_en_route",
        "day1_added_miles": day1_added_miles,
        "day1_added_drive": day1_added_drive,
        "day1_added_wait": 0.0,
        "day1_final_drive": drive_used + day1_added_drive,
        "day1_final_duty": duty_used + day1_added_drive,
        "break_start_time": break_start_time,
        "break_end_time": break_end_time,
        "day2_arrival_B": arrival_B_day2,
        "service_start_B": service_start_B,
        "day2_drive_used_before_service": remaining_hours_to_B_after_break,
        "day2_duty_used_before_service": remaining_hours_to_B_after_break + day2_wait_at_B,
        "day2_added_miles": remaining_hours_to_B_after_break * DRIVING_SPEED,
        "day2_added_wait": day2_wait_at_B,
    }


def evaluate_overnight_route(day1_route, day2_route, verbose=False):
    if not day1_route or not day2_route:
        return {"overall_feasible": False, "reason": "Both day1_route and day2_route must be nonempty"}

    first_zip_day1 = day1_route[0]["TOZIP"]
    first_drive_day1 = get_distance(DEPOT_ZIP, first_zip_day1) / DRIVING_SPEED
    dispatch_time = WINDOW_OPEN - first_drive_day1

    if verbose:
        print("Dispatch day 1:", to_clock(dispatch_time))

    seg1 = serve_route_segment(
        day1_route,
        start_zip=DEPOT_ZIP,
        start_time=dispatch_time,
        drive_used=0,
        duty_used=0,
        verbose=verbose,
    )

    first_zip_day2 = day2_route[0]["TOZIP"]
    trans = latest_break_transition(
        last_zip=seg1["end_zip"],
        first_next_zip=first_zip_day2,
        finish_time=seg1["end_time"],
        drive_used=seg1["total_drive"],
        duty_used=seg1["total_duty"],
    )

    if not trans["feasible"]:
        return {"overall_feasible": False, "reason": trans["reason"]}

    if verbose:
        print("\nOvernight transition")
        print(" Case:", trans["case"])
        print(" Break starts:", to_clock(trans["break_start_time"]))
        print(" Break ends:", to_clock(trans["break_end_time"]))
        print(" Service at first day-2 stop starts:", to_clock(trans["service_start_B"]))

    seg2 = serve_route_segment(
        day2_route,
        start_zip=first_zip_day2,
        start_time=trans["service_start_B"],
        drive_used=trans["day2_drive_used_before_service"],
        duty_used=trans["day2_duty_used_before_service"],
        verbose=verbose,
    )

    miles_back = get_distance(seg2["end_zip"], DEPOT_ZIP)
    drive_back = miles_back / DRIVING_SPEED
    return_time = seg2["end_time"] + drive_back

    total_miles = (
        seg1["total_miles"]
        + trans["day1_added_miles"]
        + trans["day2_added_miles"]
        + seg2["total_miles"]
        + miles_back
    )

    total_wait = (
        seg1["total_wait"]
        + trans["day1_added_wait"]
        + trans["day2_added_wait"]
        + seg2["total_wait"]
    )

    total_unload = seg1["total_unload"] + seg2["total_unload"]
    total_cube = seg1["total_cube"] + seg2["total_cube"]

    day1_drive = trans["day1_final_drive"]
    day1_duty = trans["day1_final_duty"]

    day2_drive = seg2["total_drive"] + drive_back
    day2_duty = seg2["total_duty"] + drive_back

    capacity_feasible = total_cube <= VAN_CAPACITY
    window_feasible = seg1["window_feasible"] and seg2["window_feasible"]
    day1_dot_feasible = day1_drive <= MAX_DRIVING and day1_duty <= MAX_DUTY
    day2_dot_feasible = day2_drive <= MAX_DRIVING and day2_duty <= MAX_DUTY
    dot_feasible = day1_dot_feasible and day2_dot_feasible
    overall_feasible = capacity_feasible and window_feasible and dot_feasible

    results = {
        "total_miles": int(total_miles),
        "total_unload": round(float(total_unload), 3),
        "total_wait": round(float(total_wait), 3),
        "total_cube": int(total_cube),
        "day1_drive": round(float(day1_drive), 3),
        "day1_duty": round(float(day1_duty), 3),
        "day2_drive": round(float(day2_drive), 3),
        "day2_duty": round(float(day2_duty), 3),
        "break_start_time": round(float(trans["break_start_time"]), 3),
        "break_end_time": round(float(trans["break_end_time"]), 3),
        "return_time": round(float(return_time), 3),
        "capacity_feasible": bool(capacity_feasible),
        "window_feasible": bool(window_feasible),
        "day1_dot_feasible": bool(day1_dot_feasible),
        "day2_dot_feasible": bool(day2_dot_feasible),
        "dot_feasible": bool(dot_feasible),
        "overall_feasible": bool(overall_feasible),
    }

    if verbose:
        print("\nReturn to depot:", to_clock(return_time))
        print("\nTotals")
        print(results)

    return results


# =========================
# Construction / improvement
# =========================
def compare_overnight_pair(route1, route2):
    separate_miles = evaluate_route(route1, verbose=False)["total_miles"] + evaluate_route(route2, verbose=False)["total_miles"]
    overnight = evaluate_overnight_route(route1, route2, verbose=False)

    if overnight["overall_feasible"] and overnight["total_miles"] < separate_miles:
        return {
            "improves": True,
            "separate_miles": separate_miles,
            "overnight_miles": overnight["total_miles"],
            "savings": separate_miles - overnight["total_miles"],
            "overnight_results": overnight,
        }

    return {"improves": False, "separate_miles": separate_miles}


def best_insertion(route_list, candidate_orders):
    best_order = None
    best_route = None
    best_extra_miles = float("inf")

    base_miles = evaluate_route(route_list, verbose=False)["total_miles"]

    for _, cand in candidate_orders.iterrows():
        for pos in range(len(route_list) + 1):
            trial_route = route_list[:pos] + [cand] + route_list[pos:]
            results = evaluate_route(trial_route, verbose=False)

            if results["overall_feasible"]:
                extra_miles = results["total_miles"] - base_miles
                if extra_miles < best_extra_miles:
                    best_order = cand
                    best_route = trial_route
                    best_extra_miles = extra_miles

    return best_order, best_route


def build_route_from_seed(day_orders, seed):
    route_list = [seed]
    remaining = day_orders[day_orders["ORDERID"] != seed["ORDERID"]].copy()

    while True:
        best_order, best_route = best_insertion(route_list, remaining)
        if best_order is None:
            break
        route_list = best_route
        remaining = remaining[remaining["ORDERID"] != best_order["ORDERID"]].copy()

    return route_list, remaining


def build_one_route(day_orders):
    candidate_seeds = [
        day_orders.iloc[0],
        day_orders.loc[day_orders["TOZIP"].apply(lambda z: get_distance(DEPOT_ZIP, z)).idxmax()],
        day_orders.loc[day_orders["CUBE"].idxmax()],
    ]

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
        score = evaluate_route(route_list, verbose=False)["total_miles"]

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
    route_infos = []
    for i, route in enumerate(all_routes):
        results = evaluate_route(route, verbose=False)
        route_infos.append((i, results["total_cube"], results["total_miles"]))

    route_infos.sort(key=lambda x: (x[1], x[2]))

    for remove_idx, _, _ in route_infos:
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
    best_results = evaluate_route(best_route, verbose=False)
    improved = True

    while improved:
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 2, len(best_route)):
                trial_route = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
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
    return [two_opt_route(route) if len(route) >= 4 else route for route in all_routes]


def find_best_overnight_candidate(routes_day1, routes_day2):
    best = None
    best_savings = 0

    for i, r1 in enumerate(routes_day1):
        for j, r2 in enumerate(routes_day2):
            comp = compare_overnight_pair(r1, r2)

            if comp["improves"] and comp["savings"] > best_savings:
                best_savings = comp["savings"]
                best = {
                    "route1_idx": i,
                    "route2_idx": j,
                    "savings": comp["savings"],
                    "separate_miles": comp["separate_miles"],
                    "overnight_miles": comp["overnight_miles"],
                    "overnight_results": comp["overnight_results"],
                }

    return best


def apply_overnight_improvements(routes_by_day):
    overnight_routes = []
    used_routes = {day: set() for day in routes_by_day}

    for d1, d2 in ADJACENT_DAY_PAIRS:
        best = find_best_overnight_candidate(routes_by_day[d1], routes_by_day[d2])
        if best is None:
            continue

        i = best["route1_idx"]
        j = best["route2_idx"]

        if i in used_routes[d1] or j in used_routes[d2]:
            continue

        used_routes[d1].add(i)
        used_routes[d2].add(j)

        overnight_routes.append({
            "day1": d1,
            "day2": d2,
            "route1_idx": i,
            "route2_idx": j,
            "route1": routes_by_day[d1][i],
            "route2": routes_by_day[d2][j],
            "results": best["overnight_results"],
            "savings": best["savings"],
        })

    return overnight_routes, used_routes


# =========================
# Reporting
# =========================
def route_summary(route):
    r = evaluate_route(route, verbose=False)
    return {
        "orders": route_ids(route),
        "miles": r["total_miles"],
        "cube": r["total_cube"],
        "duty": r["total_duty"],
        "feasible": r["overall_feasible"],
    }


def print_day_report(day, routes):
    print(f"\n===== {day} =====")
    day_total_miles = 0

    for i, route in enumerate(routes, start=1):
        s = route_summary(route)
        print(
            f"Route {i}: {s['orders']} | "
            f"miles={s['miles']} | cube={s['cube']} | "
            f"duty={s['duty']} | feasible={s['feasible']}"
        )
        day_total_miles += s["miles"]

    print(f"{day} route count: {len(routes)}")
    print(f"{day} total miles: {day_total_miles}")
    return day_total_miles


def print_overnight_checks(routes_by_day):
    for d1, d2 in ADJACENT_DAY_PAIRS:
        best = find_best_overnight_candidate(routes_by_day[d1], routes_by_day[d2])

        print(f"\nOvernight check: {d1} -> {d2}")
        if best is None:
            print("No improving overnight pair found.")
        else:
            print(
                f"Best pair: {d1} Route {best['route1_idx'] + 1} + "
                f"{d2} Route {best['route2_idx'] + 1} | "
                f"separate={best['separate_miles']} | "
                f"overnight={best['overnight_miles']} | "
                f"savings={best['savings']}"
            )


def print_applied_overnights(overnight_routes):
    print("\n===== APPLIED OVERNIGHT ROUTES =====")
    if not overnight_routes:
        print("No overnight routes applied.")
        return

    for k, ovn in enumerate(overnight_routes, start=1):
        print(
            f"Overnight {k}: {ovn['day1']} Route {ovn['route1_idx'] + 1} "
            f"+ {ovn['day2']} Route {ovn['route2_idx'] + 1} | "
            f"day1_orders={route_ids(ovn['route1'])} | "
            f"day2_orders={route_ids(ovn['route2'])} | "
            f"savings={ovn['savings']} | "
            f"miles={ovn['results']['total_miles']} | "
            f"feasible={ovn['results']['overall_feasible']}"
        )


# =========================
# Main
# =========================
def main():
    routes_by_day = {}

    for day in DAYS:
        day_orders = ORDERS[ORDERS["DayOfWeek"] == day].copy()
        routes = build_all_routes_for_day(day_orders)
        routes = consolidate_routes(routes)
        routes = improve_routes_by_2opt(routes)
        routes_by_day[day] = routes

    weekly_total_miles = 0
    weekly_total_routes = 0

    for day in DAYS:
        weekly_total_miles += print_day_report(day, routes_by_day[day])
        weekly_total_routes += len(routes_by_day[day])

    print("\n===== WEEKLY SUMMARY =====")
    print("Total weekly routes:", weekly_total_routes)
    print("Total weekly miles:", weekly_total_miles)

    print_overnight_checks(routes_by_day)

    overnight_routes, used_routes = apply_overnight_improvements(routes_by_day)
    print_applied_overnights(overnight_routes)

    final_total_miles = 0
    final_total_routes = 0

    for day in DAYS:
        for idx, route in enumerate(routes_by_day[day]):
            if idx not in used_routes[day]:
                final_total_miles += evaluate_route(route, verbose=False)["total_miles"]
                final_total_routes += 1

    for ovn in overnight_routes:
        final_total_miles += ovn["results"]["total_miles"]
        final_total_routes += 1

    print("\n===== FINAL SUMMARY WITH OVERNIGHT =====")
    print("Total routes:", final_total_routes)
    print("Total miles:", final_total_miles)


ORDERS, DIST_MATRIX = load_inputs()

if __name__ == "__main__":
    main()