import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from soundscape_psd_functions import median_data, station_low_noise_med_per_freq

supp = True
OUTPUT_csv = "data/stations_dist2river.csv"

# Load station data
seismo_data = pd.read_csv(OUTPUT_csv)
seismo_data['Station_num'] = pd.to_numeric(seismo_data['Station'], errors='coerce')
mask = (
    (seismo_data['Station_num'] < 1500) &
    seismo_data['dist_to_stream_km'].notna()
)
selected = seismo_data.loc[mask]

# stations as strings (matching later use) and distances as numeric array
stations = selected['Station_num'].astype(int).astype(str).values
distances = selected['dist_to_stream_km'].values

# Step 1: Parallel XML parsing
station_freqs = {}
station_powers = {}
dist_bounds = [0.3]

with ProcessPoolExecutor() as executor:
    for station, freqs, powers in executor.map(station_low_noise_med_per_freq, stations):
        station_freqs[station] = freqs
        station_powers[station] = powers

aligned_powers = np.zeros((len(stations), 125))  # will expand columns
aligned_powers_med = np.zeros((0, 125))  # will expand columns
for i, station in enumerate(stations):
    station = str(station)
    f, p = median_data(station_freqs[station], station_powers[station])
    aligned_powers[i,:] = p
    if distances[i] > dist_bounds[-1]:
        aligned_powers_med = np.vstack((aligned_powers_med, p))
med_power_all = np.median(aligned_powers_med, axis=0)

#Plot the aligned powers minus the median power for distance ranges
fig, ax = plt.subplots(figsize=(8, 6))
power_diff_1 = np.zeros((0, 125))
power_diff_over = np.zeros((0, 125))

for i, distance_km in enumerate(distances):
    power_stack = aligned_powers[i] - med_power_all
    #take median of powers for stations bellow 0.35 km, between 0.35km and 0.55, and above 0.55
    if distance_km < dist_bounds[0]:
        power_diff_1 = np.vstack((power_diff_1, power_stack))
    elif distance_km >= dist_bounds[0]:
        power_diff_over = np.vstack((power_diff_over, power_stack))

groups = [power_diff_1, power_diff_over]
colors = ['#377eb8', '#ff7f00']

labels = ['< ' + str(dist_bounds[0]) + ' km', '>= ' + str(dist_bounds[0]) + ' km']

# choose a reference frequency array from the loaded station_freqs (fallback to any available)
if len(stations) > 0 and stations[0] in station_freqs:
    freq_ref = station_freqs[stations[0]]
else:
    # fallback to the first entry in station_freqs
    freq_ref = next(iter(station_freqs.values()))

for h, c, l in zip(groups, colors, labels):
    power_med = np.median(h, axis=0)
    power_mad = np.median(np.abs(h - power_med), axis=0)
    ax.plot(freq_ref, power_med, color=c, linewidth=1.5, label=f"{l} [{len(h)} sensors]")
    if supp:
        ax.fill_between(freq_ref, power_med - power_mad, power_med + power_mad, color=c, alpha=0.1, zorder=10)
        ax.plot(freq_ref, power_med - power_mad, color=c, linestyle='dashed', linewidth=0.7)
        ax.plot(freq_ref, power_med + power_mad, color=c, linestyle='dashed', linewidth=0.7)

ax.set_xscale("log")
ax.set_xlabel("Frequency, Hz", fontsize='large')
ax.set_ylabel("Power Difference, dB", fontsize='large')
ax.grid()
ax.set_xlim(5.2556e-3, 2.4355e2)
# Place legend inside the plot at top-left
ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize='large')
fig.tight_layout()
plt.savefig("river_signal_near_road.pdf")