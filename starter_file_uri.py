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
print("\nCleaned distance matrix:")
print(dist_matrix.shape)
print(dist_matrix.head())

order_zips = set(orders["TOZIP"].unique())
distance_zips = set(dist_matrix.index)
