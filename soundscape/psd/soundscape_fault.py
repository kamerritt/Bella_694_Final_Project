import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from soundscape_psd_functions import load_stations_for_distance_calculation, parse_xml_file, median_data, align_and_median_psd, station_low_noise_med_per_freq


seismo_data = load_stations_for_distance_calculation(fullstations=True)
station = seismo_data['Station']
lat = seismo_data['Latitude']

# Select stations >= 1500 and their distances to Nenana (exclude NaNs)
seismo_data['Station'] = pd.to_numeric(seismo_data['Station'], errors='coerce')
mask = (
    (seismo_data['Station'] >= 1500) &
    (seismo_data['Station'] <= 1595) 
)
selected = seismo_data.loc[mask]


stations = selected['Station'].astype(int).astype(str).values
latitude = selected['Latitude'].astype(float).values

color = np.zeros((len(stations), 125))
station_freqs = {}
station_powers = {}
x = []
y = []
listst = 0
low = True
for i, station in enumerate(stations):
    station = str(station)
    if not low:
        sta, frequencies, powers = parse_xml_file(station,channel="Z")
        f, p = median_data(frequencies, powers)
    else:
        sta, f, p = station_low_noise_med_per_freq(station)
    station_freqs[station] = f
    station_powers[station] = p

    listst += 1
    y.append(latitude[i])
    for j in range(len(f)):
        if i == 0:
            x.append(f[j])
        color[i,j] = p[j]
aligned_powers, median_power = align_and_median_psd(station_freqs, station_powers, stations)

global_low_noise_median = np.nanmedian(aligned_powers, axis=0)
color = aligned_powers - global_low_noise_median

plt.figure()
# set 0 to be white
max_abs = np.max(np.abs(color))
plt.pcolormesh(x, y, color, vmin=-max_abs, vmax=max_abs, cmap='seismic') 
idx = np.arange(0, len(stations), 1)
tick_positions = []
for i in idx:
    tick_positions.append(latitude[i])
tick_labels = []
first_station = stations[0]
last_station = stations[-1]

for p in idx:
    if stations[p] == first_station:
        tick_labels.append(str(first_station))
    elif stations[p] == last_station:
        tick_labels.append(str(last_station))
    elif str(stations[p]) == "1557":
        tick_labels.append("Denali Fault")
    elif str(stations[p]) == "1521":
        tick_labels.append("Valley")
    else:
        tick_labels.append("")

# Remove tick marks that don't have an associated label (empty string)
filtered = [(p, l) for p, l in zip(tick_positions, tick_labels) if l != ""]
if filtered:
    tick_positions, tick_labels = zip(*filtered)
    tick_positions = np.array(tick_positions)
    tick_labels = list(tick_labels)
else:
    tick_positions = np.array([])
    tick_labels = []
plt.yticks(tick_positions, tick_labels)
plt.axvline(x=0.1, color='k', linestyle='--', linewidth=1)
plt.axvline(x=1, color='k', linestyle='--', linewidth=1)
plt.axvline(x=10, color='k', linestyle='--', linewidth=1)
plt.axvline(x=100, color='k', linestyle='--', linewidth=1)
   
plt.xscale('log')
plt.xlabel('Frequency, Hz', fontsize='x-large')
plt.colorbar(label='Power Difference, dB', fraction=0.02, pad=0.01, aspect=50)

plt.show()

