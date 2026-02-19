import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from soundscape_psd_functions import median_data, parse_xml_file

OUTPUT_csv = "data/stations_dist2road.csv"
seismo_data = pd.read_csv(OUTPUT_csv)

stations = seismo_data['Station'].astype(str).values
distances = seismo_data['dist_to_ROAD_km'].values

power_diff_1 = np.zeros((0, 125))
power_diff_3 = np.zeros((0, 125))

for i, station in enumerate(stations):
    station = str(station)
    _, freqs, powers = parse_xml_file(station, "Z")
    f, p = median_data(freqs, powers)
    if distances[i] < 0.1 or int(station) == 1304:
        power_diff_1 = np.vstack((power_diff_1, p))
    elif distances[i] >= 0.1:
        power_diff_3 = np.vstack((power_diff_3, p))

med_power_all = np.median(power_diff_3, axis=0)
freqs = f

color = ['#377eb8', '#ff7f00']
labels = ['< 0.1 km', '>= 0.1 km']

fig, ax = plt.subplots(2,1, figsize=(6, 10), sharex=True)

for h, c, l in zip([power_diff_1,power_diff_3], color, labels):
    power_med = np.median(h, axis=0)
    power_mad = np.median(np.abs(h - power_med), axis=0)

    ax[0].plot(freqs, power_med, color=c, linewidth=1, label=l + "[" + str(len(h)) + " sensors]", zorder=5)
    ax[0].fill_between(freqs, power_med - power_mad, power_med + power_mad, color=c, alpha=0.15, zorder=2)
    ax[0].plot(freqs, power_med - power_mad, color=c, linestyle='dashed', linewidth=0.7, zorder=5)
    ax[0].plot(freqs, power_med + power_mad, color=c, linestyle='dashed', linewidth=0.7, zorder=5)

    power_med = power_med - med_power_all
    ax[1].plot(freqs, power_med, color=c, linewidth=1, label=l + "[" + str(len(h)) + " sensors]", zorder=5)
    ax[1].fill_between(freqs, power_med - power_mad, power_med + power_mad, color=c, alpha=0.15, zorder=2)
    ax[1].plot(freqs, power_med - power_mad, color=c, linestyle='dashed', linewidth=0.7, zorder=5)
    ax[1].plot(freqs, power_med + power_mad, color=c, linestyle='dashed', linewidth=0.7, zorder=5)

ax[1].set_xlabel("Frequency, Hz", fontsize='x-large')
ax[0].set_ylabel("Power, dB", fontsize='x-large')
ax[1].set_ylabel("Power Difference, dB", fontsize='x-large')
ax[1].legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize='large')
for axs in ax.flat:
    axs.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    axs.set_xlim(5.2556e-3, 2.4355e2)
    axs.set_xscale('log')
    axs.set_axisbelow(True)
    axs.tick_params(axis='both', which='both', labelsize='large')
fig.tight_layout()
plt.show()
