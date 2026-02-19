import numpy as np
import pandas as pd
from matplotlib import cm, colors

from matplotlib import pyplot as plt
from soundscape_psd_functions import station_low_noise_med_per_freq


supp = False
OUTPUT_csv = "data/stations_dist2river.csv"


# Load station data
seismo_data = pd.read_csv(OUTPUT_csv)

# Select stations >= 1500 and their distances to Nenana (exclude NaNs)
seismo_data['Station_num'] = pd.to_numeric(seismo_data['Station'], errors='coerce')
mask = (
    (seismo_data['Station_num'] >= 1500) &
    (seismo_data['Station_num'] <= 1590) &
    seismo_data['dist_to_nenana_km'].notna()
)
selected = seismo_data.loc[mask]

# stations as strings (matching later use) and distances as numeric array
stations = selected['Station_num'].astype(int).astype(str).values
distances = selected['dist_to_nenana_km'].values

# Step 1: Parallel XML parsing
station_freqs = {}
station_powers = {}

for i, station in enumerate(stations):
    station = str(station)
    sta, f, p = station_low_noise_med_per_freq(station)
    station_freqs[station] = f
    station_powers[station] = p 

aligned_powers = np.zeros((len(stations), 125))  # will expand columns
aligned_powers_med = np.zeros((0, 125))  # will expand columns
for i, station in enumerate(stations):
    station = str(station)
    aligned_powers[i,:] = station_powers[station]
    if distances[i] > 0.46:
        aligned_powers_med = np.vstack((aligned_powers_med, station_powers[station]))
med_power_all = np.median(aligned_powers_med, axis=0)

# Plot the aligned powers colored by distance to river
fig, ax = plt.subplots()

freq_ref = station_freqs[stations[0]]

# Set up colormap based on distance
norm = colors.Normalize(vmin=np.min(distances), vmax=np.max(distances))
cmap = cm.plasma

for row, dist in zip(aligned_powers, distances):
    power_diff = row #- med_power_all
    ax.plot(
        freq_ref,
        power_diff,
        color=cmap(norm(dist)),
        linewidth=1.5
    )

# Colorbar
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Distance to Nenana River (km)")

ax.set_xscale("log")
ax.set_xlabel("Frequency (Hz)", fontsize='large')
ax.set_ylabel("Power Difference [10log10(m**2/sec**4/Hz)][dB]", fontsize='large')
ax.grid()
ax.set_xlim(5.2556e-3, 2.4355e2)

fig.tight_layout()
plt.show()