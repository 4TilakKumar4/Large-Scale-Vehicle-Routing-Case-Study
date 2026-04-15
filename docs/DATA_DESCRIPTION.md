# Data Description
### IE 7200 — Supply Chain Engineering | NHG Vehicle Routing Case Study

---

## 1. Source

The dataset is provided as supplemental material to the *Growing Pains* teaching case (Milburn, Kirac, and Hadianniasar, 2017). It describes an average week of freight requirements for Northeastern Home Goods (NHG), representative of mean weekly volumes over the prior year. Store addresses are aggregated to ZIP code level to protect the identity of the original client.

The distance matrix was generated using the Google Maps API in calendar year 2014 and reflects road-network distances rounded to the nearest integer mile.

---

## 2. Raw Input Files

### 2.1 `deliveries.xlsx` — two sheets

**Sheet: `OrderTable`**

One row per store delivery order. The depot row (ORDERID = 0) is excluded during cleaning.

| Column | Type | Description |
|---|---|---|
| `ORDERID` | Integer | Unique order identifier. 0 = depot (excluded). |
| `FROMZIP` | Integer | Origin ZIP code. Always 01887 (Wilmington depot). |
| `TOZIP` | Integer | Destination store ZIP code. |
| `CUBE` | Float | Order volume in cubic feet. May contain comma-formatted values (e.g. `2,699`) — stripped during cleaning. |
| `DayOfWeek` | String | Scheduled delivery day: Mon, Tue, Wed, Thu, or Fri. |
| `ST required?` | String | Whether the order requires a Straight Truck: `yes` or `no`. Renamed to `straight_truck_required` during cleaning. |

**Sheet: `LocationTable`**

One row per unique ZIP code appearing in the dataset (stores + depot).

| Column | Type | Description |
|---|---|---|
| `ZIP` | Integer | 5-digit ZIP code (stored without leading zero). |
| `X` | Float | Longitude of ZIP code centroid. Renamed to `lon` during cleaning. |
| `Y` | Float | Latitude of ZIP code centroid. Renamed to `lat` during cleaning. |
| `CITY` | String | City name. |
| `STATE` | String | Two-letter state abbreviation. |
| `ZIPID` | Integer | Sequential internal ZIP identifier used as distance matrix column/row index. |

### 2.2 `distances.xlsx` — sheet `Sheet1`

A 124×124 matrix of road-network distances in miles between all location pairs (depot + 123 store ZIPs). Rows and columns are indexed by ZIP code. The first two columns (`Unnamed: 0`, `Unnamed: 1`) are the ZIP code and ZIPID respectively; the remaining 123 columns are destination ZIP codes.

---

## 3. Processed Files (written by `VRP_DataAnalysis.py`)

All solver scripts read from these files. They are written to `data/` and are not committed to the repository.

### 3.1 `data/orders_clean.csv`

Cleaned version of `OrderTable`. Depot row removed, CUBE values parsed, types coerced.

| Column | Type | Description |
|---|---|---|
| `ORDERID` | Float | Order identifier |
| `FROMZIP` | Float | Always 1887 |
| `TOZIP` | Float | Destination store ZIP |
| `CUBE` | Float | Order volume (ft³) |
| `DayOfWeek` | String | Mon / Tue / Wed / Thu / Fri |
| `straight_truck_required` | String | yes / no |

### 3.2 `data/locations_clean.csv`

Cleaned version of `LocationTable`.

| Column | Type | Description |
|---|---|---|
| `ZIP` | Float | ZIP code |
| `lon` | Float | Longitude |
| `lat` | Float | Latitude |
| `CITY` | String | City name |
| `STATE` | String | State abbreviation |
| `ZIPID` | Float | Internal sequential ID |

### 3.3 `data/distance_matrix.csv`

Square distance matrix, indexed and columned by ZIP code integer. Loaded by `vrp_solvers/base.py::loadInputs()`.

---

## 4. Dataset Statistics

### 4.1 Orders

| Metric | Value |
|---|---|
| Total orders (excl. depot) | 262 |
| Unique store ZIP codes | 123 |
| States served | CT, MA, ME, NH, NY, RI, VT |
| Depot ZIP | 01887 (Wilmington, MA) |

### 4.2 Weekly freight requirements

| Day | Orders | Total Cube (ft³) | Mean Cube (ft³) |
|---|---|---|---|
| Monday | 43 | 10,223 | ~238 |
| Tuesday | 58 | 11,537 | ~199 |
| Wednesday | 50 | 15,192 | ~304 |
| Thursday | 63 | 15,009 | ~238 |
| Friday | 47 | 13,468 | ~287 |
| **Total** | **262** | **65,429** | **~250** |

### 4.3 Store delivery frequency

| Deliveries per week | Number of stores |
|---|---|
| 1 | 78 |
| 2 | 25 |
| 3 | 9 |
| 4 | 10 |
| 5 | 1 |

### 4.4 Distance matrix

| Metric | Value |
|---|---|
| ZIPs in matrix | 124 (depot + 123 stores) |
| Matrix dimensions | 124 × 124 |
| Distance units | Integer miles (road network) |
| Source | Google Maps API, calendar year 2014 |

---

## 5. Key Data Notes

**CUBE formatting** — raw values in `OrderTable` may include commas as thousands separators (e.g. `2,699`) and surrounding whitespace. `VRP_DataAnalysis.py::loadAndCleanOrders()` strips these before numeric conversion. The raw file should not be modified.

**ZIP code storage** — ZIP codes are stored as integers without leading zeros. ZIP 01887 is stored as 1887. All distance matrix lookups use integer keys. Geocoding via `pgeocode` requires zero-padded 5-digit strings (re-added during geocoding in `VRP_BaseCase_Map.py`).

**Distance matrix column type** — when loaded from CSV, column headers are initially strings. `loadInputs()` coerces them to numeric using `pd.to_numeric(..., errors="coerce")` to ensure integer ZIP lookups resolve correctly.

**Straight truck column** — the `straight_truck_required` column is present in `orders_clean.csv` and is used in Sub-problem 2 (mixed fleet). In the base case (Sub-problem 1) it is ignored; all orders are assigned to vans.

---

## 6. Reference

Milburn, A.B., Kirac, E., and Hadianniasar, M. (2017). Growing Pains: A Case Study for Large-Scale Vehicle Routing. *INFORMS Transactions on Education*, 17(2), 81–84. https://doi.org/10.1287/ited.2016.0167cs
