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

# pick two orders from the same day
same_day_orders = orders[orders["DayOfWeek"] == "Wed"].copy()
print(same_day_orders.head())

order1 = same_day_orders.iloc[0]
order2 = same_day_orders.iloc[1]
order3 = same_day_orders.iloc[3]

route_list = [order1 , order2, order3]

current_zip = depot_zip
total_miles = 0
total_drive = 0
total_unload = 0
total_wait = 0
total_cube = 0

first_zip = route_list[0]["TOZIP"]
first_drive = get_distance(depot_zip, first_zip) / driving_speed
current_time = window_open - first_drive
print("Dispatch:", to_clock(current_time))

all_before_close = True

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

print("\nReturn to depot:", to_clock(return_time))

print("\nTotals")
print("Total miles:", total_miles)
print("Total drive hours:", total_drive)
print("Total unload hours:", total_unload)
print("Total wait hours:", total_wait)
print("Total duty hours:", total_duty)
print("Total cube:", total_cube)

print("\nConstraints")
print("Capacity feasible:", total_cube <= van_capacity)
print("Window feasible:", all_before_close)
print("DOT feasible:", total_drive <= max_driving and total_duty <= max_duty)