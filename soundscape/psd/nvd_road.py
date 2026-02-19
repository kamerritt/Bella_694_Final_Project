import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from matplotlib import pyplot as plt
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from soundscape_psd_functions import median_data

def psds_day(station, channel="Z", day=True):
    xml_file = f"psd_stations/psd_{station}_DP{channel}.xml"
    tree = ET.parse(xml_file)
    root = tree.getroot()
    freqs = []
    powers = []
    for psd in root.findall(".//Psd"):
        start_time_str = psd.get("start")
        start_dt = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        hour_key = start_dt.hour
        if day:
            if 5 <= int(hour_key) <= 16:
                continue
        if not day:
            if 16 < int(hour_key) or int(hour_key) < 5:
                continue
        # Extract all freq/power pairs for this PSD
        for value in psd.findall("value"):
            freqs.append(float(value.get("freq")))
            powers.append(float(value.get("power")))
    return station, np.array(freqs), np.array(powers)

OUTPUT_csv = "stations_dist2road.csv"
seismo_data = pd.read_csv(OUTPUT_csv)
over_1500 = True
if over_1500:
    # Select stations >= 1500 and specific stations 1161-1164, keep only rows with a Nenana distance
    seismo_data['Station_num'] = pd.to_numeric(seismo_data['Station'], errors='coerce')
    keep_list = [1161, 1162, 1163, 1164]
    mask = ((seismo_data['Station_num'] >= 1500) | (seismo_data['Station_num'].isin(keep_list))) & seismo_data['dist_to_ROAD_km'].notna()
    selected = seismo_data.loc[mask].copy()

    # stations as strings (matching later use) and distances as numeric array
    stations = selected['Station_num'].astype(int).astype(str).values
    distances = selected['dist_to_ROAD_km'].values
else:
    stations = seismo_data['Station'].astype(str).values
    distances = seismo_data['dist_to_ROAD_km'].values
days = [False, True, False, True]

fig, ax = plt.subplots(2,2, sharex=True)
for tt, d in enumerate(days):
    station_freqs = {}
    station_powers = {}
    with ProcessPoolExecutor() as executor:
        for station, freqs, powers in executor.map(psds_day, stations, [ "2"] * len(stations), [d] * len(stations)):
            f, p = median_data(freqs, powers)
            station_freqs[station] = f
            station_powers[station] = p 

    aligned_powers = np.zeros((len(stations), 125)) 
    power_grid_no_values_less_than_01 = np.zeros((1, 125))
    for i, station in enumerate(stations):
        station = str(station)
        f, p = median_data(station_freqs[station], station_powers[station])
        aligned_powers[i,:] = p
        if distances[i] >= 0.3:
            power_grid_no_values_less_than_01 = np.vstack([power_grid_no_values_less_than_01, p])
    if d == False:
        med_power_all = np.median(power_grid_no_values_less_than_01, axis=0)
    power_diff_1 = np.zeros((0, 125))
    power_diff_3 = np.zeros((0, 125))
    for i, distance_km in enumerate(distances):
        if tt in [0, 1]:
            power_stack = aligned_powers[i]
        else:
            power_stack = aligned_powers[i] - med_power_all
        if distance_km < 0.1:
            power_diff_1 = np.vstack((power_diff_1, power_stack))
        elif distance_km >= 0.3:
            power_diff_3 = np.vstack((power_diff_3, power_stack))

    color = ['#377eb8', '#ff7f00', '#4daf4a',
                    '#f781bf', '#a65628', '#984ea3',
                    '#999999', '#e41a1c', '#dede00']

    labels = ['< 0.1 km', '>= 0.3 km']
    for h, c, l in zip([power_diff_1,power_diff_3], color, labels):

        power_med = np.median(h, axis=0)
        power_mad = np.median(np.abs(h - power_med), axis=0)
        if tt < 2:
            ax[0,tt].set_xscale('log')
            ax[0,tt].set_axisbelow(True)

            ax[0,tt].plot(station_freqs['1500'], power_med, color=c, linewidth=1, label=l + "[" + str(len(h)) + " sensors]", zorder=5)
            ax[0,tt].fill_between(station_freqs['1500'], power_med - power_mad, power_med + power_mad, color=c, alpha=0.15, zorder=2)
            ax[0,tt].plot(station_freqs['1500'], power_med - power_mad, color=c, linestyle='dashed', linewidth=0.7, zorder=5)
            ax[0,tt].plot(station_freqs['1500'], power_med + power_mad, color=c, linestyle='dashed', linewidth=0.7, zorder=5)
            ax[0,tt].set_title("Daytime" if d else "Nighttime")
            ax[0,tt].grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax[0,tt].set_xlim(5.2556e-3, 2.4355e2)
            if tt == 0:
                ax[0,tt].set_ylabel("Power")
        else:
            ax[1,tt-2].set_xscale('log')
            ax[1,tt-2].set_axisbelow(True)

            ax[1,tt-2].plot(station_freqs['1500'], power_med, color=c, linewidth=1, label=l + "[" + str(len(h)) + " sensors]", zorder=5)
            ax[1,tt-2].fill_between(station_freqs['1500'], power_med - power_mad, power_med + power_mad, color=c, alpha=0.15, zorder=2)
            ax[1,tt-2].plot(station_freqs['1500'], power_med - power_mad, color=c, linestyle='dashed', linewidth=0.7, zorder=5)
            ax[1,tt-2].plot(station_freqs['1500'], power_med + power_mad, color=c, linestyle='dashed', linewidth=0.7, zorder=5)
            ax[1,tt-2].set_xlabel("Frequency (Hz)")
            if tt == 3:
                ax[1,tt-2].legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize='large')
            if tt == 2:
                ax[1,tt-2].set_ylabel("Power Difference")
            ax[1,tt-2].grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax[1,tt-2].set_xlim(5.2556e-3, 2.4355e2)


fig.tight_layout()
plt.show()