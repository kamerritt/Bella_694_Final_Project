import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib.colors import TwoSlopeNorm
from soundscape_psd_functions import parse_psds_grouped_by_day, median_data


OUTPUT_DIR = "daily_psd_colormaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_median_psd(psds):
    all_freqs = np.unique(np.concatenate([f for f, _ in psds]))

    aligned = np.vstack([
        np.interp(all_freqs, f, p)
        for f, p in psds
    ])

    return all_freqs, np.median(aligned, axis=0)


seismo_data = pd.read_csv('/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/parkshwy_nodes.txt', sep="|")
stations = seismo_data["Station"].astype(str).values

# PARSE STATIONS
daily_station_psd = defaultdict(dict)

for station in stations:
    xml_file = f"psd_stations/psd_{station}.xml"
    if not os.path.exists(xml_file):
        continue

    daily_psds = parse_psds_grouped_by_day(xml_file)

    for day, psds in daily_psds.items():
        if len(psds) < 2:
            continue

        freqs, med_power = compute_median_psd(psds)
        daily_station_psd[day][station] = (freqs, med_power)

# GLOBAL COLOR SCALE 
all_values = []

for day, station_data in daily_station_psd.items():
    freqs_all = np.unique(np.concatenate([v[0] for v in station_data.values()]))

    powers = []
    for freqs, p in station_data.values():
        powers.append(np.interp(freqs_all, freqs, p))

    powers = np.array(powers)
    powers -= np.median(powers, axis=0)  # network median (valid stations only)
    all_values.append(powers)

all_values = np.concatenate(all_values)
vmax = np.nanpercentile(np.abs(all_values), 99)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

# ---------------- PLOT BY DAY ----------------
for day, station_data in daily_station_psd.items():

    freqs_all = np.unique(np.concatenate([v[0] for v in station_data.values()]))
    color = np.full((len(stations), len(freqs_all)), np.nan)

    # --- fill only stations with data ---
    valid_rows = []
    for i, station in enumerate(stations):
        if station not in station_data:
            continue

        freqs, powers = station_data[station]
        color[i, :] = np.interp(freqs_all, freqs, powers)
        valid_rows.append(i)

    # --- remove median using ONLY valid stations ---
    color[valid_rows, :] -= np.median(color[valid_rows, :], axis=0)

    # ---------------- PLOT ----------------
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(
        freqs_all,
        np.arange(len(stations)),
        color,
        cmap="seismic",
        norm=norm,
        shading="auto"
    )
    # show every 10th station starting at the first (use actual station names)
    tick_positions = np.arange(0, len(stations), 10)
    tick_labels = [str(stations[p]) for p in tick_positions]
    plt.yticks(tick_positions, tick_labels)

    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Station")
    plt.title(f"Daily Median PSD (Network Median Removed)\n{day}")

    plt.colorbar(label="Relative Median Power (0 = Network Median)")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, f"daily_psd_colormap_{day}.png"),
        dpi=200
    )
    plt.close()

print(f"Saved daily color-scale PSD plots to {OUTPUT_DIR}")
