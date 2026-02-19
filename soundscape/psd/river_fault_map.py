import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pyproj import Transformer
from shapely.geometry import LineString
import numpy as np

# -------------------------
# PATHS AND CONSTANTS
# -------------------------
STATIONS_txt = "/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/full_nodes.txt"
GDB_PATH = "data/river_data/hydrusm010g.gdb"
STREAM_LAYER = "Stream"
FAULT_SHP = "data/DenaliF_shp/DenaliF_spbtrace.shp"

LAT_COL = "Latitude"
LON_COL = "Longitude"
TARGET_CRS = "EPSG:3338"   # Alaska Albers Equal Area (meters)

# Distance group definitions
bins = [0, 0.25, 0.32, 0.39, 0.46, float("inf")]
labels = ['< 0.25 km', '0.25 - 0.32 km', '0.32 - 0.39 km',
          '0.39 - 0.46 km', '>= 0.46km']
colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#999999']

# -------------------------
# READ FAULT
# -------------------------
fault = gpd.read_file(FAULT_SHP).to_crs(TARGET_CRS)

# -------------------------
# READ STATIONS
# -------------------------
stations_df = pd.read_csv(STATIONS_txt, sep="|")

stations = gpd.GeoDataFrame(
    stations_df,
    geometry=gpd.points_from_xy(
        stations_df[LON_COL],
        stations_df[LAT_COL]
    ),
    crs="EPSG:4326"
).to_crs(TARGET_CRS)

stations["Station_num"] = pd.to_numeric(
    stations["Station"],
    errors="coerce"
)

# Select stations >= 1500
stations_sel = stations[stations["Station_num"] >= 1500].copy()

stations_color = stations_sel[stations_sel["Station_num"] < 1591].copy()
stations_black = stations_sel[stations_sel["Station_num"] >= 1591].copy()

# -------------------------
# READ STREAMS
# -------------------------
streams = gpd.read_file(GDB_PATH, layer=STREAM_LAYER).to_crs(TARGET_CRS)

name_fields = [c for c in streams.columns if "name" in c.lower()]
if not name_fields:
    raise ValueError("No stream name field found")

name_field = name_fields[0]

nenana = streams[
    streams[name_field].str.contains("Nenana", case=False, na=False)
].dissolve()

# -------------------------
# DISTANCE TO RIVER (km)
# -------------------------
stations_color["dist_km"] = (
    stations_color.geometry.distance(nenana.geometry.iloc[0]) / 1000.0
)

stations_color["dist_group"] = pd.cut(
    stations_color["dist_km"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# -------------------------
# CREATE PERPENDICULAR LINE
# -------------------------
station_1521 = stations.loc[stations["Station_num"] == 1521]
station_1522 = stations.loc[stations["Station_num"] == 1522]

p1 = station_1521.geometry.iloc[0]
p2 = station_1522.geometry.iloc[0]

# Direction vector between stations
dx = p2.x - p1.x
dy = p2.y - p1.y

# Midpoint
mx = (p1.x + p2.x) / 2
my = (p1.y + p2.y) / 2

# Perpendicular direction
perp_dx = -dy
perp_dy = dx

# Normalize
length = np.sqrt(perp_dx**2 + perp_dy**2)
perp_dx /= length
perp_dy /= length

# Length of perpendicular segment (meters)
half_length = 50  # 50 m on each side (adjust if desired)

x1 = mx - perp_dx * half_length
y1 = my - perp_dy * half_length
x2 = mx + perp_dx * half_length
y2 = my + perp_dy * half_length

perp_line = LineString([(x1, y1), (x2, y2)])
perp_line_gdf = gpd.GeoDataFrame(geometry=[perp_line], crs=TARGET_CRS)

# -------------------------
# COMPUTE ZOOM EXTENT
# -------------------------
xmin = 258.5 * 1000
xmax = 259.4 * 1000
ymin = 1507.4 * 1000
ymax = 1509.2 * 1000
print(stations_sel[["Station", "Station_num"]].sort_values("Station_num"))
print(stations_black[["Station", "Station_num"]])

# -------------------------
# PLOTTING
# -------------------------
fig, ax = plt.subplots()

nenana.plot(ax=ax, color="blue", linewidth=3)
fault.plot(ax=ax, color="black", linewidth=1)

# Colored stations (<1591)
for label, color in zip(labels, colors):
    subset = stations_color[stations_color["dist_group"] == label]
    if not subset.empty:
        subset.plot(ax=ax, color=color, markersize=25)

# Black stations (>=1591)
if not stations_black.empty:
    stations_black.plot(ax=ax, color="black", markersize=25)

# Plot perpendicular line
if perp_line_gdf is not None:
    perp_line_gdf.plot(ax=ax, color="black", linewidth=1)

# Zoom
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")

# -------------------------
# AXIS TICKS (lat/lon labels)
# -------------------------
transformer = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)

xmid = (xmin + xmax) / 2.0
ymid = (ymin + ymax) / 2.0

def x_to_lon(x, pos):
    lon, lat = transformer.transform(x, ymid)
    return f"{lon:.3f}"

def y_to_lat(y, pos):
    lon, lat = transformer.transform(xmid, y)
    return f"{lat:.3f}"

def _show_every_other_tick(x, pos, axis):
    ticks = ax.get_xticks() if axis == "x" else ax.get_yticks()
    if len(ticks) == 0:
        return x_to_lon(x, pos) if axis == "x" else y_to_lat(x, pos)
    idx = min(range(len(ticks)), key=lambda i: abs(ticks[i] - x))
    if idx % 2 == 1:
        return ""
    return x_to_lon(x, pos) if axis == "x" else y_to_lat(x, pos)

ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: _show_every_other_tick(x, pos, "x"))
)
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: _show_every_other_tick(x, pos, "y"))
)

ax.set_xlabel("Longitude", fontsize='x-large')
ax.set_ylabel("Latitude", fontsize='x-large')

ax.tick_params(direction="out", length=6, width=1)
ax.grid()

plt.tight_layout()
plt.show()
