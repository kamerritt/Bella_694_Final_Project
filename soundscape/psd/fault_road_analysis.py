import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from soundscape_psd_functions import parse_xml_file, median_data

OUTPUT_csv = "data/stations_dist2road.csv"
seismo_data = pd.read_csv(OUTPUT_csv)

# Convert station column to numeric
seismo_data['Station_num'] = pd.to_numeric(
    seismo_data['Station'], errors='coerce'
)

keep_list = [1161, 1162, 1163, 1164]

# --------------------------------------------------
# Define station groups
# --------------------------------------------------

# Blue group (left column): stations < 1500
mask_lt1500 = (
    (seismo_data['Station_num'] < 1500) &
    seismo_data['dist_to_ROAD_km'].notna()
)
data_lt1500 = seismo_data.loc[mask_lt1500].copy()

# Blue group (right column): keep_list only
mask_keep = (
    (seismo_data['Station_num'].isin(keep_list)) &
    seismo_data['dist_to_ROAD_km'].notna()
)
data_keep = seismo_data.loc[mask_keep].copy()

# Orange group (same everywhere): stations >= 1500
mask_ge1500 = (
    (seismo_data['Station_num'] >= 1500) &
    seismo_data['dist_to_ROAD_km'].notna()
)
data_ge1500 = seismo_data.loc[mask_ge1500].copy()

groups_blue = [data_lt1500, data_keep]

# --------------------------------------------------
# Cache PSDs (parse each station once)
# --------------------------------------------------

all_station_nums = seismo_data['Station_num'].dropna().unique()

psd_cache = {}
freqs = None

for station in all_station_nums:
    station_str = str(int(station))
    _, f_raw, powers = parse_xml_file(station_str, "Z")
    f, p = median_data(f_raw, powers)

    psd_cache[station] = p
    freqs = f

# --------------------------------------------------
# Build ORANGE reference group (>=1500) once
# --------------------------------------------------

orange_stack = []

for _, row in data_ge1500.iterrows():
    station = row['Station_num']
    orange_stack.append(psd_cache[station])

orange_stack = np.array(orange_stack)

orange_med = np.median(orange_stack, axis=0)
orange_mad = np.median(np.abs(orange_stack - orange_med), axis=0)

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(2, 2, figsize=(6, 10),
                       sharex=True)

blue_color = '#377eb8'
orange_color = '#ff7f00'

for col, group in enumerate(groups_blue):

    blue_stack = []

    for _, row in group.iterrows():
        station = row['Station_num']
        blue_stack.append(psd_cache[station])

    blue_stack = np.array(blue_stack)

    if len(blue_stack) == 0:
        continue

    blue_med = np.median(blue_stack, axis=0)
    blue_mad = np.median(np.abs(blue_stack - blue_med), axis=0)

    # --------------------
    # TOP ROW (absolute)
    # --------------------

    # Blue
    ax[0, col].plot(freqs, blue_med,
                    color=blue_color, linewidth=1)

    ax[0, col].fill_between(freqs,
                            blue_med - blue_mad,
                            blue_med + blue_mad,
                            color=blue_color, alpha=0.15)

    # Orange (same on both columns)
    ax[0, col].plot(freqs, orange_med,
                    color=orange_color, linewidth=1)

    ax[0, col].fill_between(freqs,
                            orange_med - orange_mad,
                            orange_med + orange_mad,
                            color=orange_color, alpha=0.15)
    ax[0, col].set_ylim(-160, -90)
    # --------------------
    # BOTTOM ROW (difference)
    # --------------------

    blue_diff = blue_med - orange_med
    orange_diff = orange_med - orange_med  # zero line for reference
    ax[1, col].plot(freqs, blue_diff,
                    color=blue_color, linewidth=1,
                    label=f"< 1 km [{len(blue_stack)}]")

    ax[1, col].fill_between(freqs,
                            blue_diff - blue_mad,
                            blue_diff + blue_mad,
                            color=blue_color, alpha=0.15)
    
    ax[1, col].plot(freqs, orange_diff,
                    color=orange_color, linewidth=1,
                    label=f">=1 km [{len(orange_stack)}]")

    ax[1, col].fill_between(freqs,
                            orange_diff - orange_mad,
                            orange_diff + orange_mad,
                            color=orange_color, alpha=0.15)

    ax[1, col].axhline(0, color='k', linewidth=0.8)
    ax[1, col].set_ylim(-6, 45)
    ax[1, col].legend(loc='upper left',
                        bbox_to_anchor=(0.02, 0.98),
                        frameon=True,
                        fontsize='large')
# remove the tick marks on the right column
ax[0, 1].tick_params(right=False)
ax[1, 1].tick_params(right=False)   
# --------------------------------------------------
# Formatting
# --------------------------------------------------

ax[1, 0].set_xlabel("Frequency, Hz", fontsize='x-large')
ax[1, 1].set_xlabel("Frequency, Hz", fontsize='x-large')
ax[0, 0].set_ylabel("Power, dB", fontsize='x-large')
ax[1, 0].set_ylabel("Power Difference, dB", fontsize='x-large')
# remove 0 black line the bottom row
ax[1, 0].axhline(0, color=orange_color, zorder=10)
ax[1, 1].axhline(0, color=orange_color, zorder=10)

for axs in ax.flat:
    axs.grid(which='both', linestyle='--',
             linewidth=0.5, alpha=0.7)
    axs.set_xlim(5.2556e-3, 2.4355e2)
    axs.set_xscale('log')
    axs.set_axisbelow(True)

fig.tight_layout()
plt.show()
