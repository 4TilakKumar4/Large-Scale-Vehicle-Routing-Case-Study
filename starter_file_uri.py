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


# grab one example order
sample_order = orders.iloc[0]

print("\nSample order:")
print(sample_order)

order_id = sample_order["ORDERID"]
dest_zip = sample_order["TOZIP"]
cube = sample_order["CUBE"]
day = sample_order["DayOfWeek"]

print("\nParsed values:")
print("Order ID:", order_id)
print("Destination ZIP:", dest_zip)
print("Cube:", cube)
print("Day:", day)

# distance and drive time from depot to this order
miles = get_distance(depot_zip, dest_zip)
drive_time = miles / driving_speed

print("\nTravel info:")
print("Miles from depot:", miles)
print("Drive time (hours):", drive_time)

# unload time for this order
unload_time = max(min_time, unload_rate * cube)

print("\nService info:")
print("Unload time (minutes):", unload_time)
print("Unload time (hours):", unload_time / 60)