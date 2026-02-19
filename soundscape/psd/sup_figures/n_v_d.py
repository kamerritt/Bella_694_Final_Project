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

# compute day-night differences and associated MADs (handle missing groups)
def diff_and_mad(m_day, m_night, mad_day, mad_night):
    diff = m_day - m_night
    combined = np.sqrt(mad_day**2 + mad_night**2)
    return diff, combined

# helper to compute median and MAD for a specific station list
def group_stats(station_list):
    aligned = np.zeros((len(station_list), 125))
    for i, st in enumerate(station_list):
        _, p = median_data(station_freqs[st], station_powers[st])
        aligned[i, :] = p
    med = np.median(aligned, axis=0)
    mad = np.median(np.abs(aligned - med), axis=0)
    return med, mad

OUTPUT_csv = "stations_dist2road.csv"
seismo_data = pd.read_csv(OUTPUT_csv)

stations = seismo_data['Station'].astype(str).values
distances = seismo_data['dist_to_ROAD_km'].values
station_to_dist = {str(s): d for s, d in zip(stations, distances)}

# Define two groups: <0.1 km and >0.3 km
group_near = [s for s, d in station_to_dist.items() if d < 0.1]
group_far = [s for s, d in station_to_dist.items() if d > 0.3]

days = [False, True]

fig, ax = plt.subplots(1, 1)

for d in days:
    station_freqs = {}
    station_powers = {}
    # compute per-station median PSD for this time selection
    with ProcessPoolExecutor() as executor:
        for station, freqs, powers in executor.map(psds_day, stations, ["Z"] * len(stations), [d] * len(stations)):
            f, p = median_data(freqs, powers)
            station_freqs[station] = f
            station_powers[station] = p

    med_near, mad_near = group_stats(group_near)
    med_far, mad_far = group_stats(group_far)

    if d:
        med_day_near, mad_day_near = med_near, mad_near
        med_day_far, mad_day_far = med_far, mad_far
    else:
        med_night_near, mad_night_near = med_near, mad_near
        med_night_far, mad_night_far = med_far, mad_far

diff_near, comb_near = diff_and_mad(med_day_near, med_night_near, mad_day_near, mad_night_near)
diff_far, comb_far = diff_and_mad(med_day_far, med_night_far, mad_day_far, mad_night_far)

ax.set_xscale('log')
ax.set_axisbelow(True)

ax.plot(station_freqs['1500'], diff_near, linewidth=1, color='#377eb8', label='< 0.1 km from road', zorder=5)
#ax.fill_between(station_freqs['1500'], diff_near - comb_near, diff_near + comb_near, color='#377eb8', alpha=0.15, zorder=2)
#ax.plot(station_freqs['1500'], diff_near - comb_near, linestyle='dashed', color='#377eb8', linewidth=0.7, zorder=5)
#ax.plot(station_freqs['1500'], diff_near + comb_near, linestyle='dashed', color='#377eb8', linewidth=0.7, zorder=5)

ax.plot(station_freqs['1500'], diff_far, linewidth=1, color= '#ff7f00', label='> 0.3 km from road', zorder=5)
#ax.fill_between(station_freqs['1500'], diff_far - comb_far, diff_far + comb_far, color='#ff7f00', alpha=0.15, zorder=2)
#ax.plot(station_freqs['1500'], diff_far - comb_far, linestyle='dashed', color='#ff7f00', linewidth=0.7, zorder=5)
#ax.plot(station_freqs['1500'], diff_far + comb_far, linestyle='dashed', color='#ff7f00', linewidth=0.7, zorder=5)
ax.axhline(0, color='black', linewidth=1, zorder=-1)
ax.set_xlabel("Frequency (Hz)", fontsize='large')
ax.set_ylabel("Power Difference [10log10(m**2/sec**4/Hz)][dB]", fontsize='large')
ax.set_title("Daytime - Nighttime PSDs (near vs far from road)", fontsize='large')
ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize='large')
ax.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_xlim(5.2556e-3, 2.4355e2)
fig.tight_layout()
plt.show()