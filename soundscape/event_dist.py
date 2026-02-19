# ===============================
# Distance Calculations Using UTM
# ===============================

import math
import pyproj
import numpy as np
import pandas as pd

# ---------------------------------
# UTM Projection (Zone 6, WGS84)
# ---------------------------------
utm_proj = pyproj.Proj(proj="utm", zone=6, ellps="WGS84")


# ---------------------------------
# Helper Functions
# ---------------------------------
def latlon_to_utm(lat, lon):
    """Convert latitude/longitude to UTM (meters)."""
    return utm_proj(lon, lat)


def utm_distance_km(x1, y1, x2, y2):
    """Euclidean distance in km between two UTM points."""
    return math.hypot(x2 - x1, y2 - y1) / 1000.0


def station_utm_distance_km(st_lat, st_lon, pt_lat, pt_lon):
    """Distance (km) between two lat/lon points using UTM."""
    x1, y1 = latlon_to_utm(st_lat, st_lon)
    x2, y2 = latlon_to_utm(pt_lat, pt_lon)
    return utm_distance_km(x1, y1, x2, y2)

# ==========================================
# Distance from Specific Station to Point
# ==========================================
def print_station_distance(station_id, target_lat, target_lon):
    mask = stations == str(station_id)
    if not mask.any():
        print(f"Station {station_id} not found.")
        return

    row = seismo_data.loc[mask].iloc[0]
    st_lat = float(row["Latitude"])
    st_lon = float(row["Longitude"])

    dist_km = station_utm_distance_km(st_lat, st_lon, target_lat, target_lon)

    print(f"UTM distance: {dist_km:.3f} km \n")

# ==========================================
# Manual Mining Blast Distance Checks
# ==========================================
print("Station 1230 distance (km) to explosion (UTM):")
print("B1:", station_utm_distance_km(63.982154, -149.121513, 63.98, -148.66))
print("B2:", station_utm_distance_km(63.982154, -149.121513, 63.97, -148.68))
print("B3:", station_utm_distance_km(63.4056, -148.8602, 64.01, -148.76))


# ==========================================
# Load Seismometer Data
# ==========================================
seismo_data = pd.read_csv(
    "/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/parkshwy_nodes.txt",
    sep="|"
)

stations = seismo_data["Station"].astype(str)
seismo_lat = seismo_data["Latitude"].astype(float)
seismo_lon = seismo_data["Longitude"].astype(float)

# Convert all stations to UTM
seismo_utm = np.array(
    [latlon_to_utm(lat, lon) for lat, lon in zip(seismo_lat, seismo_lon)]
)
seismo_x = seismo_utm[:, 0]
seismo_y = seismo_utm[:, 1]


# ==========================================
# Distance to Blast Location
# ==========================================
blast_lat = 63.9901
blast_lon = -148.7392
blast_x, blast_y = latlon_to_utm(blast_lat, blast_lon)

# Vectorized distance computation
dists_km = np.hypot(seismo_x - blast_x, seismo_y - blast_y) / 1000.0

min_idx = int(np.argmin(dists_km))
print(f"Closest station to blast: {stations.iloc[min_idx]}, "
      f"Distance: {dists_km[min_idx]:.3f} km \n")


# Station 1248 to blast
print(f"Distance from blast to station 1248:")
print_station_distance("1248", blast_lat, blast_lon)

# Station 1245 to given earthquake epicenter coordinates
olon = -149.93
olat = 61.46
print(f"Distance from epicenter to station 1245:")
print_station_distance("1245", olat, olon)

# ==========================================
# Distance from Station 1127 to Railroad
# ==========================================
rail_data = pd.read_csv(
    "/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/Alaska_Railroad.txt",
    sep=","
)

rail_lat = rail_data["Latitude"].astype(float)
rail_lon = rail_data["Longitude"].astype(float)

# Get station 1127
mask = stations == "1127"
row = seismo_data.loc[mask].iloc[0]
st_lat = float(row["Latitude"])
st_lon = float(row["Longitude"])
st_x, st_y = latlon_to_utm(st_lat, st_lon)

# Convert railroad points to UTM
rail_utm = np.array(
    [latlon_to_utm(lat, lon) for lat, lon in zip(rail_lat, rail_lon)]
)

rail_x = rail_utm[:, 0]
rail_y = rail_utm[:, 1]

# Compute distances
rail_dists_km = np.hypot(rail_x - st_x, rail_y - st_y) / 1000.0

min_idx = int(np.argmin(rail_dists_km))
print(f"Closest railroad distance to station 1127:")
print(f"Coordinate: ({rail_lat.iloc[min_idx]}, {rail_lon.iloc[min_idx]})")
print(f"Distance: {rail_dists_km[min_idx]:.3f} km, {rail_dists_km[min_idx] * 1000:.0f} m")

# ==========================================
# Distance from Station 1119 to Railroad
# ==========================================
# Get station 1119
mask = stations == "1119"
row = seismo_data.loc[mask].iloc[0]
st_lat = float(row["Latitude"])
st_lon = float(row["Longitude"])
st_x, st_y = latlon_to_utm(st_lat, st_lon)

# Convert railroad points to UTM
rail_utm = np.array(
    [latlon_to_utm(lat, lon) for lat, lon in zip(rail_lat, rail_lon)]
)

rail_x = rail_utm[:, 0]
rail_y = rail_utm[:, 1]

# Compute distances
rail_dists_km = np.hypot(rail_x - st_x, rail_y - st_y) / 1000.0

min_idx = int(np.argmin(rail_dists_km))
print(f"Closest railroad distance to station 1119:")
print(f"Coordinate: ({rail_lat.iloc[min_idx]}, {rail_lon.iloc[min_idx]})")
print(f"Distance: {rail_dists_km[min_idx]:.3f} km, {rail_dists_km[min_idx] * 1000:.0f} m")

# ==========================================
# Distance between Station 1245 to Road
# ==========================================
road_data = pd.read_csv(
    "/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/parks_highway.txt",
    sep=","
)

road_lat = road_data["Latitude"].astype(float)
road_lon = road_data["Longitude"].astype(float)

# Get station 1245
mask = stations == "1245"
row = seismo_data.loc[mask].iloc[0]
st_lat = float(row["Latitude"])
st_lon = float(row["Longitude"])
st_x, st_y = latlon_to_utm(st_lat, st_lon)

# Convert road points to UTM
road_utm = np.array(
    [latlon_to_utm(lat, lon) for lat, lon in zip(road_lat, road_lon)]
)

road_x = road_utm[:, 0]
road_y = road_utm[:, 1]

# Compute distances
road_dists_km = np.hypot(road_x - st_x, road_y - st_y) / 1000.0

min_idx = int(np.argmin(road_dists_km))
print(f"Closest road distance to station 1245:")
print(f"Coordinate: ({road_lat.iloc[min_idx]}, {road_lon.iloc[min_idx]})")
print(f"Distance: {road_dists_km[min_idx]:.3f} km, {road_dists_km[min_idx] * 1000:.0f} m")
