import numpy as np
from matplotlib import pyplot as plt
from soundscape_psd_functions import load_stations_for_distance_calculation

fault = True
lat_check = True
seismic_sta = load_stations_for_distance_calculation(fullstations=True)
elevation = seismic_sta['Elevation']
station = seismic_sta['Station']
lat = seismic_sta['Latitude']

plt.figure(figsize=(10, 4))
plt.scatter(lat, elevation, c='k', marker='o')
if not lat_check:
    if not fault:
        plt.xlim(1000, 1307)
        plt.xticks(np.arange(1000, 1308, 10))
    else:
        plt.xlim(1500, 1591)
        plt.xticks(np.arange(1500, 1592, 10))
else:
    for lat, elevation, station in zip(lat, elevation, station):
        plt.text(lat, elevation, str(station))

plt.grid()
plt.show()
