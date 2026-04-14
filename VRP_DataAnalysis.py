"""
VRP_DataAnalysis.py — Data cleaning, EDA, and processed data export for the NHG VRP project.

Run this script first. It reads raw deliveries.xlsx and distances.xlsx,
cleans both, exports three CSVs to data/, and writes all EDA plots and
summary CSVs to outputs/eda/.
"""

import os
import warnings

import folium
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from folium.plugins import HeatMap
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
EDA_DIR    = os.path.join(BASE_DIR, "outputs", "eda")
RAW_ORDERS = os.path.join(BASE_DIR, "deliveries.xlsx")
RAW_DIST   = os.path.join(BASE_DIR, "distances.xlsx")

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

PLOT_STYLE = {
    "figsize":    (10, 6),
    "dpi":        150,
    "fontSize":   11,
    "titleSize":  13,
    "colorMain":  "#457B9D",
    "colorSeq":   ["#E63946", "#F4A261", "#2A9D8F", "#457B9D", "#9B5DE5"],
    "gridAlpha":  0.3,
    "spineColor": "#CCCCCC",
}

plt.rcParams.update({
    "font.size":         PLOT_STYLE["fontSize"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        PLOT_STYLE["gridAlpha"],
    "grid.color":        PLOT_STYLE["spineColor"],
})


def makeDirs():
    """Create data/ and outputs/eda/ if they do not already exist."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(EDA_DIR,  exist_ok=True)
    except OSError as e:
        raise OSError(
            f"Could not create required directories: {e}\n"
            "Check that you have write permissions in the project root."
        )


def saveFig(name):
    path = os.path.join(EDA_DIR, name)
    plt.savefig(path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def loadAndCleanOrders():
    """Read OrderTable, strip comma-formatted cubes, drop depot row, coerce types."""
    if not os.path.exists(RAW_ORDERS):
        raise FileNotFoundError(
            f"deliveries.xlsx not found at {RAW_ORDERS}\n"
            "Place the raw data files in the project root before running this script."
        )

    try:
        orders = pd.read_excel(RAW_ORDERS, sheet_name="OrderTable")
    except ValueError:
        raise ValueError(
            "Could not find sheet 'OrderTable' in deliveries.xlsx. "
            "Check that the file has not been renamed or restructured."
        )

    # CUBE column can arrive as ' 2,699 ' with embedded commas and whitespace
    orders["CUBE"] = (
        orders["CUBE"].astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    orders = orders[orders["ORDERID"] != 0].copy()
    orders = orders[orders["CUBE"] != "(Depot)"].copy()

    orders["CUBE"]    = pd.to_numeric(orders["CUBE"],    errors="coerce")
    orders["FROMZIP"] = pd.to_numeric(orders["FROMZIP"], errors="coerce")
    orders["TOZIP"]   = pd.to_numeric(orders["TOZIP"],   errors="coerce")
    orders["ORDERID"] = pd.to_numeric(orders["ORDERID"], errors="coerce")
    orders = orders.dropna(subset=["CUBE", "TOZIP", "ORDERID"])

    if orders.empty:
        raise ValueError(
            "OrderTable cleaned to zero rows. "
            "Check that deliveries.xlsx contains valid order data."
        )

    if "ST required?" in orders.columns:
        orders = orders.rename(columns={"ST required?": "straight_truck_required"})
        orders["straight_truck_required"] = orders["straight_truck_required"].fillna("no")

    orders = orders[["ORDERID", "FROMZIP", "TOZIP", "CUBE", "DayOfWeek",
                      "straight_truck_required"]].copy()
    orders = orders.reset_index(drop=True)
    return orders


def loadAndCleanLocations():
    """Read LocationTable from the Excel file; rename X/Y to lon/lat."""
    # deliveries.xlsx existence already checked in loadAndCleanOrders()
    try:
        locs = pd.read_excel(RAW_ORDERS, sheet_name="LocationTable")
    except ValueError:
        raise ValueError(
            "Could not find sheet 'LocationTable' in deliveries.xlsx. "
            "Check that the file has not been renamed or restructured."
        )

    locs = locs.rename(columns={"X": "lon", "Y": "lat"})
    locs["ZIP"]   = pd.to_numeric(locs["ZIP"],   errors="coerce")
    locs["ZIPID"] = pd.to_numeric(locs["ZIPID"], errors="coerce")
    locs = locs.dropna(subset=["ZIP"])

    if locs.empty:
        raise ValueError(
            "LocationTable cleaned to zero rows. "
            "Check that deliveries.xlsx contains valid location data."
        )

    return locs


def loadAndCleanDistances():
    """Read distances.xlsx; return a square DataFrame indexed and columned by ZIP int."""
    if not os.path.exists(RAW_DIST):
        raise FileNotFoundError(
            f"distances.xlsx not found at {RAW_DIST}\n"
            "Place the raw data files in the project root before running this script."
        )

    try:
        distances = pd.read_excel(RAW_DIST, sheet_name="Sheet1")
    except ValueError:
        raise ValueError(
            "Could not find sheet 'Sheet1' in distances.xlsx. "
            "Check that the file has not been renamed or restructured."
        )

    distances = distances.rename(columns={"Unnamed: 0": "ZIP", "Unnamed: 1": "ZIPID"})
    distances = distances[distances["ZIP"] != "Zip"].copy()
    distances["ZIP"]   = pd.to_numeric(distances["ZIP"],   errors="coerce")
    distances["ZIPID"] = pd.to_numeric(distances["ZIPID"], errors="coerce")
    distances = distances.dropna(subset=["ZIP"])
    distances = distances.set_index("ZIP")

    distMatrix = distances.drop(columns=["ZIPID"]).copy()
    distMatrix.columns = pd.to_numeric(distMatrix.columns, errors="coerce")

    if distMatrix.empty:
        raise ValueError(
            "distances.xlsx cleaned to an empty matrix. "
            "Check that the file contains valid distance data."
        )

    return distMatrix


def exportCleanData(orders, locs, distMatrix):
    """Write cleaned data to data/ as CSV."""
    ordersPath = os.path.join(DATA_DIR, "orders_clean.csv")
    locsPath   = os.path.join(DATA_DIR, "locations_clean.csv")
    distPath   = os.path.join(DATA_DIR, "distance_matrix.csv")

    try:
        orders.to_csv(ordersPath,    index=False)
        locs.to_csv(locsPath,        index=False)
        distMatrix.to_csv(distPath)
    except OSError as e:
        raise OSError(
            f"Failed to write cleaned data to {DATA_DIR}: {e}\n"
            "Check that you have write permissions in the project root."
        )

    print(f"  Exported: {ordersPath}")
    print(f"  Exported: {locsPath}")
    print(f"  Exported: {distPath}")


def exportSummaryCsvs(orders, locs):
    """Write eda_summary.csv and daily_stats.csv to outputs/eda/."""
    storeOrders = orders.groupby("TOZIP").agg(
        orders_per_week=("ORDERID", "count"),
        total_cube=("CUBE", "sum"),
        delivery_days=("DayOfWeek", lambda x: "/".join(sorted(set(x)))),
    ).reset_index()
    storeOrders = storeOrders.merge(
        locs[["ZIP", "CITY", "STATE"]].rename(columns={"ZIP": "TOZIP"}),
        on="TOZIP", how="left"
    )
    storeOrders = storeOrders[["TOZIP", "CITY", "STATE",
                                "orders_per_week", "total_cube", "delivery_days"]]
    storeOrders.to_csv(os.path.join(EDA_DIR, "eda_summary.csv"), index=False)

    dailyStats = orders.groupby("DayOfWeek").agg(
        order_count=("ORDERID", "count"),
        total_cube=("CUBE", "sum"),
        mean_cube=("CUBE", "mean"),
        min_cube=("CUBE", "min"),
        max_cube=("CUBE", "max"),
    ).reindex(DAYS).reset_index()
    dailyStats.columns = ["day", "order_count", "total_cube",
                          "mean_cube", "min_cube", "max_cube"]
    dailyStats = dailyStats.round(1)
    dailyStats.to_csv(os.path.join(EDA_DIR, "daily_stats.csv"), index=False)

    print(f"  Saved: {os.path.join(EDA_DIR, 'eda_summary.csv')}")
    print(f"  Saved: {os.path.join(EDA_DIR, 'daily_stats.csv')}")


def plotDemandByDay(orders):
    """Bar chart of order count and total cube per weekday."""
    dayGroups = orders.groupby("DayOfWeek").agg(
        order_count=("ORDERID", "count"),
        total_cube=("CUBE", "sum"),
    ).reindex(DAYS)

    fig, ax1 = plt.subplots(figsize=PLOT_STYLE["figsize"])
    x     = np.arange(len(DAYS))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, dayGroups["order_count"],
                    width, label="Orders", color=PLOT_STYLE["colorSeq"][0], alpha=0.85)
    ax1.set_ylabel("Order Count")
    ax1.set_xlabel("Day of Week")
    ax1.set_xticks(x)
    ax1.set_xticklabels(DAYS)

    ax2   = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, dayGroups["total_cube"],
                    width, label="Total Cube (ft³)", color=PLOT_STYLE["colorSeq"][3], alpha=0.85)
    ax2.set_ylabel("Total Cube (ft³)")

    lines  = [bars1, bars2]
    labels = [b.get_label() for b in lines]
    ax1.legend(lines, labels, loc="upper left")
    ax1.set_title("Daily Demand: Order Count and Total Cube", fontsize=PLOT_STYLE["titleSize"])

    saveFig("demand_by_day.png")


def plotDemandDistribution(orders):
    """Histogram of individual order cube sizes."""
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    ax.hist(orders["CUBE"], bins=30, color=PLOT_STYLE["colorMain"],
            edgecolor="white", alpha=0.85)
    ax.set_xlabel("Order Cube (ft³)")
    ax.set_ylabel("Number of Orders")
    ax.set_title("Distribution of Order Sizes", fontsize=PLOT_STYLE["titleSize"])
    ax.axvline(orders["CUBE"].mean(), color="#E63946", linestyle="--",
               linewidth=1.5, label=f"Mean: {orders['CUBE'].mean():.0f} ft³")
    ax.axvline(orders["CUBE"].median(), color="#2A9D8F", linestyle="--",
               linewidth=1.5, label=f"Median: {orders['CUBE'].median():.0f} ft³")
    ax.legend()
    saveFig("demand_distribution.png")


def plotStoreDeliveryFrequency(orders):
    """Bar chart of how many stores receive 1-5 deliveries per week."""
    freqCounts = (
        orders.groupby("TOZIP")["DayOfWeek"]
        .nunique()
        .value_counts()
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(freqCounts.index.astype(str), freqCounts.values,
           color=PLOT_STYLE["colorSeq"], edgecolor="white", alpha=0.85)
    ax.set_xlabel("Deliveries per Week")
    ax.set_ylabel("Number of Stores")
    ax.set_title("Store Delivery Frequency", fontsize=PLOT_STYLE["titleSize"])

    for i, v in enumerate(freqCounts.values):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=PLOT_STYLE["fontSize"])

    saveFig("store_delivery_frequency.png")


def plotGeographicScatter(orders, locs):
    """Scatter of store lat/lon coloured by state."""
    storeZips = orders["TOZIP"].unique()
    storeLocs = locs[locs["ZIP"].isin(storeZips)].copy()
    depotLoc  = locs[locs["ZIP"] == 1887]

    states      = storeLocs["STATE"].unique()
    stateColors = {
        s: PLOT_STYLE["colorSeq"][i % len(PLOT_STYLE["colorSeq"])]
        for i, s in enumerate(sorted(states))
    }

    fig, ax = plt.subplots(figsize=(11, 8))

    for state, group in storeLocs.groupby("STATE"):
        ax.scatter(group["lon"], group["lat"],
                   c=stateColors[state], label=state,
                   s=60, alpha=0.85, edgecolors="white", linewidths=0.5)

    if not depotLoc.empty:
        ax.scatter(depotLoc["lon"].values, depotLoc["lat"].values,
                   c="black", marker="*", s=300, zorder=5, label="Depot (Wilmington)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Store Locations by State", fontsize=PLOT_STYLE["titleSize"])
    ax.legend(title="State", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    saveFig("geographic_scatter.png")


def plotDemandHeatmapStatic(orders, locs):
    """Static matplotlib heatmap of demand intensity by ZIP centroid."""
    storeZips = orders.groupby("TOZIP")["CUBE"].sum().reset_index()
    storeZips = storeZips.merge(
        locs[["ZIP", "lat", "lon"]].rename(columns={"ZIP": "TOZIP"}),
        on="TOZIP", how="left"
    ).dropna(subset=["lat", "lon"])

    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(
        storeZips["lon"], storeZips["lat"],
        c=storeZips["CUBE"], cmap="YlOrRd",
        s=storeZips["CUBE"] / storeZips["CUBE"].max() * 400 + 30,
        alpha=0.8, edgecolors="white", linewidths=0.4
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Total Weekly Cube (ft³)")

    depotLoc = locs[locs["ZIP"] == 1887]
    if not depotLoc.empty:
        ax.scatter(depotLoc["lon"].values, depotLoc["lat"].values,
                   c="black", marker="*", s=300, zorder=5, label="Depot")
        ax.legend()

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Demand Heatmap — Total Weekly Cube by Store Location",
                 fontsize=PLOT_STYLE["titleSize"])
    plt.tight_layout()
    saveFig("demand_heatmap.png")


def plotDemandHeatmapFolium(orders, locs):
    """Interactive Folium heatmap of demand density."""
    storeZips = orders.groupby("TOZIP")["CUBE"].sum().reset_index()
    storeZips = storeZips.merge(
        locs[["ZIP", "lat", "lon"]].rename(columns={"ZIP": "TOZIP"}),
        on="TOZIP", how="left"
    ).dropna(subset=["lat", "lon"])

    # Skip the Folium map if no ZIPs have coordinates rather than producing an empty file
    if storeZips.empty:
        print("  Skipped demand_heatmap.html — no store ZIPs with valid coordinates.")
        return

    depotLoc = locs[locs["ZIP"] == 1887]
    centLat  = depotLoc["lat"].values[0] if not depotLoc.empty else 42.3
    centLon  = depotLoc["lon"].values[0] if not depotLoc.empty else -71.1

    m        = folium.Map(location=[centLat, centLon], zoom_start=8, tiles="CartoDB positron")
    heatData = [[row["lat"], row["lon"], row["CUBE"]] for _, row in storeZips.iterrows()]
    HeatMap(heatData, radius=20, blur=15, max_zoom=10).add_to(m)

    if not depotLoc.empty:
        folium.Marker(
            location=[depotLoc["lat"].values[0], depotLoc["lon"].values[0]],
            tooltip="<b>DEPOT</b> — Wilmington, MA",
            icon=folium.Icon(color="black", icon="home", prefix="fa"),
        ).add_to(m)

    path = os.path.join(EDA_DIR, "demand_heatmap.html")
    m.save(path)
    print(f"  Saved: {path}")


def plotDayWiseHeatmap(orders, locs):
    """5-panel subplot with one demand scatter per weekday."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)

    allZips = orders.groupby("TOZIP")["CUBE"].sum().reset_index()
    allZips = allZips.merge(
        locs[["ZIP", "lat", "lon"]].rename(columns={"ZIP": "TOZIP"}),
        on="TOZIP", how="left"
    ).dropna(subset=["lat", "lon"])
    vmax = allZips["CUBE"].max()

    for ax, day, color in zip(axes, DAYS, PLOT_STYLE["colorSeq"]):
        dayOrders = orders[orders["DayOfWeek"] == day]
        dayZips   = dayOrders.groupby("TOZIP")["CUBE"].sum().reset_index()
        dayZips   = dayZips.merge(
            locs[["ZIP", "lat", "lon"]].rename(columns={"ZIP": "TOZIP"}),
            on="TOZIP", how="left"
        ).dropna(subset=["lat", "lon"])

        if not dayZips.empty:
            ax.scatter(
                dayZips["lon"], dayZips["lat"],
                c=dayZips["CUBE"], cmap="YlOrRd",
                s=dayZips["CUBE"] / vmax * 200 + 20,
                vmin=0, vmax=vmax,
                alpha=0.85, edgecolors="white", linewidths=0.3
            )

        depotLoc = locs[locs["ZIP"] == 1887]
        if not depotLoc.empty:
            ax.scatter(depotLoc["lon"].values, depotLoc["lat"].values,
                       c="black", marker="*", s=150, zorder=5)

        ax.set_title(day, fontsize=PLOT_STYLE["titleSize"], color=color)
        ax.set_xlabel("Lon")
        if ax is axes[0]:
            ax.set_ylabel("Lat")

    fig.suptitle("Day-Wise Demand Heatmap", fontsize=14, y=1.02)
    plt.tight_layout()
    saveFig("day_wise_heatmap.png")


def plotDistanceMatrixHeatmap(distMatrix):
    """Heatmap of road distances between all ZIP pairs."""
    fig, ax = plt.subplots(figsize=(12, 10))

    cmap = LinearSegmentedColormap.from_list(
        "dist", ["#f0f4f8", "#457B9D", "#1d3557"]
    )
    img = ax.imshow(distMatrix.values.astype(float), cmap=cmap, aspect="auto")
    plt.colorbar(img, ax=ax, shrink=0.7, label="Distance (miles)")

    n    = len(distMatrix)
    step = max(1, n // 15)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels(distMatrix.columns[::step].astype(int), rotation=90, fontsize=7)
    ax.set_yticks(range(0, n, step))
    ax.set_yticklabels(distMatrix.index[::step].astype(int), fontsize=7)

    ax.set_title("Distance Matrix Heatmap (miles)", fontsize=PLOT_STYLE["titleSize"])
    ax.set_xlabel("Destination ZIP")
    ax.set_ylabel("Origin ZIP")
    plt.tight_layout()
    saveFig("distance_matrix_heatmap.png")


def plotCubeByState(orders, locs):
    """Horizontal bar chart of total cube demand grouped by state."""
    merged    = orders.merge(
        locs[["ZIP", "STATE"]].rename(columns={"ZIP": "TOZIP"}),
        on="TOZIP", how="left"
    )
    stateCube = merged.groupby("STATE")["CUBE"].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = [PLOT_STYLE["colorSeq"][i % len(PLOT_STYLE["colorSeq"])]
               for i in range(len(stateCube))]
    ax.barh(stateCube.index, stateCube.values, color=colors, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Total Weekly Cube (ft³)")
    ax.set_ylabel("State")
    ax.set_title("Total Weekly Demand by State", fontsize=PLOT_STYLE["titleSize"])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    saveFig("cube_by_state.png")


def plotCubePerStoreRanked(orders):
    """Bar chart of total weekly cube per store, ranked descending."""
    ordersPerStore = orders.groupby("TOZIP")["CUBE"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(ordersPerStore)), ordersPerStore.values,
           color=PLOT_STYLE["colorMain"], edgecolor="white", alpha=0.85)
    ax.set_xlabel("Store (ranked by total cube)")
    ax.set_ylabel("Total Weekly Cube (ft³)")
    ax.set_title("Weekly Cube per Store (ranked)", fontsize=PLOT_STYLE["titleSize"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    saveFig("cube_per_store_ranked.png")


def printSummary(orders, locs, distMatrix):
    nStores   = orders["TOZIP"].nunique()
    nOrders   = len(orders)
    totalCube = orders["CUBE"].sum()
    nZips     = len(distMatrix)

    print("\nData Summary")
    print("-" * 60)
    print(f"  Total orders (excl. depot): {nOrders}")
    print(f"  Unique store ZIPs:          {nStores}")
    print(f"  Total weekly cube (ft³):    {totalCube:,.0f}")
    print(f"  Distance matrix ZIPs:       {nZips}")
    print(f"  Days covered:               {orders['DayOfWeek'].unique().tolist()}")

    print("\nDaily Breakdown")
    print("-" * 60)
    for day in DAYS:
        d = orders[orders["DayOfWeek"] == day]
        print(f"  {day}: {len(d)} orders | {d['CUBE'].sum():,.0f} ft³ | "
              f"mean cube: {d['CUBE'].mean():.0f} ft³")


def main():
    print("VRP Data Analysis")
    print("=" * 60)

    print("\nCreating directories...")
    makeDirs()

    print("\nLoading and cleaning data...")
    orders     = loadAndCleanOrders()
    locs       = loadAndCleanLocations()
    distMatrix = loadAndCleanDistances()

    print("\nExporting cleaned data to data/...")
    exportCleanData(orders, locs, distMatrix)

    print("\nExporting summary CSVs to outputs/eda/...")
    exportSummaryCsvs(orders, locs)

    printSummary(orders, locs, distMatrix)

    print("\nGenerating plots...")
    plotDemandByDay(orders)
    plotDemandDistribution(orders)
    plotStoreDeliveryFrequency(orders)
    plotGeographicScatter(orders, locs)
    plotDemandHeatmapStatic(orders, locs)
    plotDemandHeatmapFolium(orders, locs)
    plotDayWiseHeatmap(orders, locs)
    plotDistanceMatrixHeatmap(distMatrix)
    plotCubeByState(orders, locs)
    plotCubePerStoreRanked(orders)

    print("\nDone. All outputs written to outputs/eda/")


if __name__ == "__main__":
    main()