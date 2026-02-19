
import pandas as pd
import numpy as np
from soundscape_psd_functions import load_stations_for_distance_calculation

stations = load_stations_for_distance_calculation(fullstations=True)

# Comparison stations
targets = {
    "PAIN": (63.73333, -148.91667),
    "PANN": (64.54796, -149.08398),
    "PATK": (62.32056, -150.09361),
}

# Haversine distance (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# Find and print closest station
for name, (lat, lon) in targets.items():
    distances = haversine(
        lat, lon,
        stations["Latitude"].values,
        stations["Longitude"].values
    )
    closest_indices = distances.argsort()[:6]
    closest_stations = stations.iloc[closest_indices]["Station"].tolist()
    
    print(f"{name} -> {closest_stations}")
