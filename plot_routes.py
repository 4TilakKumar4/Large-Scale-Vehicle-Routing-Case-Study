import pandas as pd
import folium
from itertools import cycle

import starter_file_uri as vrp


def load_location_table():
    location = pd.read_excel("deliveries.xlsx", sheet_name="LocationTable")
    location = location.dropna(subset=["ZIP", "X", "Y"]).copy()
    location["ZIP"] = location["ZIP"].astype(int)
    return location


def build_zip_lookup(location_df):
    # Y = latitude, X = longitude
    return {
        int(row["ZIP"]): (float(row["Y"]), float(row["X"]))
        for _, row in location_df.iterrows()
    }


def build_same_day_routes():
    routes_by_day = {}

    for day in vrp.DAYS:
        day_orders = vrp.ORDERS[vrp.ORDERS["DayOfWeek"] == day].copy()
        routes = vrp.build_all_routes_for_day(day_orders)
        routes = vrp.consolidate_routes(routes)
        routes = vrp.improve_routes_by_2opt(routes)
        routes_by_day[day] = routes

    return routes_by_day


def add_depot_marker(m, zip_lookup, depot_zip):
    folium.Marker(
        zip_lookup[depot_zip],
        popup=f"Depot {depot_zip}",
        icon=folium.Icon(color="black", icon="home")
    ).add_to(m)


def coords_from_zips(zip_list, zip_lookup):
    coords = []
    for z in zip_list:
        if z not in zip_lookup:
            print(f"Warning: ZIP {z} missing from LocationTable")
            return []
        coords.append(zip_lookup[z])
    return coords


def add_same_day_route(m, day, route_num, route, zip_lookup, color, depot_zip):
    route_zips = [depot_zip] + [int(stop["TOZIP"]) for stop in route] + [depot_zip]
    coords = coords_from_zips(route_zips, zip_lookup)
    if not coords:
        return

    results = vrp.evaluate_route(route, verbose=False)

    folium.PolyLine(
        coords,
        color=color,
        weight=4,
        opacity=0.8,
        popup=f"{day} Route {route_num} | miles={results['total_miles']}"
    ).add_to(m)

    for stop_idx, stop in enumerate(route, start=1):
        zip_code = int(stop["TOZIP"])
        order_id = int(stop["ORDERID"])
        cube = float(stop["CUBE"])

        folium.CircleMarker(
            location=zip_lookup[zip_code],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=(
                f"{day} Route {route_num}<br>"
                f"Stop {stop_idx}<br>"
                f"Order {order_id}<br>"
                f"ZIP {zip_code}<br>"
                f"Cube {cube}"
            )
        ).add_to(m)


def add_overnight_route(m, ovn_num, ovn, zip_lookup, color, depot_zip):
    day1 = ovn["day1"]
    day2 = ovn["day2"]
    route1 = ovn["route1"]
    route2 = ovn["route2"]
    results = ovn["results"]

    # Day 1: depot -> day1 stops
    day1_zips = [depot_zip] + [int(stop["TOZIP"]) for stop in route1]
    day1_coords = coords_from_zips(day1_zips, zip_lookup)
    if not day1_coords:
        return

    folium.PolyLine(
        day1_coords,
        color=color,
        weight=5,
        opacity=0.9,
        popup=f"Overnight {ovn_num} Day 1 ({day1}->{day2}) | miles={results['total_miles']}"
    ).add_to(m)

    # Overnight transition: last stop of day 1 -> first stop of day 2
    last_day1_zip = int(route1[-1]["TOZIP"])
    first_day2_zip = int(route2[0]["TOZIP"])
    transition_coords = coords_from_zips([last_day1_zip, first_day2_zip], zip_lookup)

    if transition_coords:
        folium.PolyLine(
            transition_coords,
            color=color,
            weight=3,
            opacity=0.7,
            dash_array="8, 10",
            popup=f"Overnight transition {day1}->{day2}"
        ).add_to(m)

    # Day 2: first day2 stop -> remaining day2 stops -> depot
    day2_zips = [int(stop["TOZIP"]) for stop in route2] + [depot_zip]
    day2_coords = coords_from_zips(day2_zips, zip_lookup)
    if not day2_coords:
        return

    folium.PolyLine(
        day2_coords,
        color=color,
        weight=5,
        opacity=0.9,
        popup=f"Overnight {ovn_num} Day 2 ({day1}->{day2}) | miles={results['total_miles']}"
    ).add_to(m)

    # Markers for day 1 stops
    for stop_idx, stop in enumerate(route1, start=1):
        zip_code = int(stop["TOZIP"])
        order_id = int(stop["ORDERID"])
        cube = float(stop["CUBE"])

        folium.CircleMarker(
            location=zip_lookup[zip_code],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=(
                f"Overnight {ovn_num}<br>"
                f"Day 1: {day1}<br>"
                f"Stop {stop_idx}<br>"
                f"Order {order_id}<br>"
                f"ZIP {zip_code}<br>"
                f"Cube {cube}"
            )
        ).add_to(m)

    # Markers for day 2 stops
    for stop_idx, stop in enumerate(route2, start=1):
        zip_code = int(stop["TOZIP"])
        order_id = int(stop["ORDERID"])
        cube = float(stop["CUBE"])

        folium.CircleMarker(
            location=zip_lookup[zip_code],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=(
                f"Overnight {ovn_num}<br>"
                f"Day 2: {day2}<br>"
                f"Stop {stop_idx}<br>"
                f"Order {order_id}<br>"
                f"ZIP {zip_code}<br>"
                f"Cube {cube}"
            )
        ).add_to(m)


def build_final_solution():
    routes_by_day = build_same_day_routes()
    overnight_routes, used_routes = vrp.apply_overnight_improvements(routes_by_day)
    return routes_by_day, overnight_routes, used_routes


def plot_final_solution(routes_by_day, overnight_routes, used_routes, zip_lookup, depot_zip=vrp.DEPOT_ZIP):
    depot_latlon = zip_lookup[depot_zip]
    m = folium.Map(location=depot_latlon, zoom_start=7)

    add_depot_marker(m, zip_lookup, depot_zip)

    same_day_colors = {
        "Mon": "red",
        "Tue": "blue",
        "Wed": "green",
        "Thu": "purple",
        "Fri": "orange",
    }

    overnight_colors = cycle([
        "black", "darkred", "darkblue", "darkgreen", "cadetblue", "pink"
    ])

    # Plot same-day routes that were NOT replaced by overnight routes
    for day in vrp.DAYS:
        route_counter = 1
        for idx, route in enumerate(routes_by_day[day]):
            if idx in used_routes[day]:
                continue

            add_same_day_route(
                m=m,
                day=day,
                route_num=route_counter,
                route=route,
                zip_lookup=zip_lookup,
                color=same_day_colors[day],
                depot_zip=depot_zip
            )
            route_counter += 1

    # Plot applied overnight routes
    for ovn_num, ovn in enumerate(overnight_routes, start=1):
        color = next(overnight_colors)
        add_overnight_route(
            m=m,
            ovn_num=ovn_num,
            ovn=ovn,
            zip_lookup=zip_lookup,
            color=color,
            depot_zip=depot_zip
        )

    return m


def print_final_summary(routes_by_day, overnight_routes, used_routes):
    final_total_miles = 0
    final_total_routes = 0

    for day in vrp.DAYS:
        for idx, route in enumerate(routes_by_day[day]):
            if idx not in used_routes[day]:
                final_total_miles += vrp.evaluate_route(route, verbose=False)["total_miles"]
                final_total_routes += 1

    for ovn in overnight_routes:
        final_total_miles += ovn["results"]["total_miles"]
        final_total_routes += 1

    print("Final routes:", final_total_routes)
    print("Final miles:", final_total_miles)


def main():
    location_df = load_location_table()
    zip_lookup = build_zip_lookup(location_df)

    routes_by_day, overnight_routes, used_routes = build_final_solution()
    print_final_summary(routes_by_day, overnight_routes, used_routes)

    final_map = plot_final_solution(routes_by_day, overnight_routes, used_routes, zip_lookup)
    final_map.save("final_overnight_solution_map.html")
    print("Saved final_overnight_solution_map.html")


if __name__ == "__main__":
    main()