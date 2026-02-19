import numpy as np
from matplotlib import pyplot as plt
from soundscape_psd_functions import (
    station_low_noise_med_per_freq,
    parse_xml_file,
    median_data,
    load_stations_for_distance_calculation,
)

# Load stations
seismic_sta = load_stations_for_distance_calculation(fullstations=True)
stations = seismic_sta["Station"].astype(str)

# Containers for PSDs
parks_low, parks = [], []
denali_low, denali = [], []

for station in stations:
    # Low-noise PSD
    _, frequencies, low_noise_power = station_low_noise_med_per_freq(station)

    # Median PSD
    _, freqs, power = parse_xml_file(station, channel="Z")
    _, median_power = median_data(freqs, power)

    # Split by region
    if int(station) <= 1306:
        parks_low.append(low_noise_power)
        parks.append(median_power)
    else:
        denali_low.append(low_noise_power)
        denali.append(median_power)

# Convert to arrays for vectorized median
parks_low = np.array(parks_low)
parks = np.array(parks)
denali_low = np.array(denali_low)
denali = np.array(denali)
assert parks.shape[1] == len(frequencies)

# Median across stations
low_noise_median_Parks = np.nanmedian(parks_low, axis=0)
low_noise_mad_Parks = np.nanmedian(np.abs(parks_low - low_noise_median_Parks), axis=0)
median_Parks = np.nanmedian(parks, axis=0)
mad_Parks = np.nanmedian(np.abs(parks - median_Parks), axis=0)

low_noise_median_Denali = np.nanmedian(denali_low, axis=0)
low_noise_mad_Denali = np.nanmedian(np.abs(denali_low - low_noise_median_Denali), axis=0)
median_Denali = np.nanmedian(denali, axis=0)
mad_Denali = np.nanmedian(np.abs(denali - median_Denali), axis=0)
colors = ['#377eb8', '#ff7f00']


fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

ax[0,0].semilogx(frequencies, median_Parks, c=colors[0])
ax[0,0].semilogx(frequencies, median_Parks+mad_Parks, c=colors[0], linestyle = '--', linewidth=0.7)
ax[0,0].semilogx(frequencies, median_Parks-mad_Parks, c=colors[0], linestyle = '--', linewidth=0.7)
ax[0,0].fill_between(frequencies, median_Parks-mad_Parks, median_Parks+mad_Parks, color=colors[0], alpha=0.2)

ax[0,1].semilogx(frequencies, low_noise_median_Parks, c=colors[0], label=f"Parks Highway [{len(parks)}]")
ax[0,1].semilogx(frequencies, low_noise_median_Parks+low_noise_mad_Parks, c=colors[0], linestyle = '--', linewidth=0.7)
ax[0,1].semilogx(frequencies, low_noise_median_Parks-low_noise_mad_Parks, c=colors[0], linestyle = '--', linewidth=0.7)
ax[0,1].fill_between(frequencies, low_noise_median_Parks-low_noise_mad_Parks, low_noise_median_Parks+low_noise_mad_Parks, color=colors[0], alpha=0.2)

ax[1,0].semilogx(frequencies, median_Denali, c=colors[1])
ax[1,0].semilogx(frequencies, median_Denali+mad_Denali, c=colors[1], linestyle = '--', linewidth=0.7)
ax[1,0].semilogx(frequencies, median_Denali-mad_Denali, c=colors[1], linestyle = '--', linewidth=0.7)
ax[1,0].fill_between(frequencies, median_Denali-mad_Denali, median_Denali+mad_Denali, color=colors[1], alpha=0.2)

ax[1,1].semilogx(frequencies, low_noise_median_Denali, c=colors[1], label=f"Denali Fault [{len(denali)}]")
ax[1,1].semilogx(frequencies, low_noise_median_Denali+low_noise_mad_Denali, c=colors[1], linestyle = '--', linewidth=0.7)
ax[1,1].semilogx(frequencies, low_noise_median_Denali-low_noise_mad_Denali, c=colors[1], linestyle = '--', linewidth=0.7)
ax[1,1].fill_between(frequencies, low_noise_median_Denali-low_noise_mad_Denali, low_noise_median_Denali+low_noise_mad_Denali, color=colors[1], alpha=0.2)

ax[0,0].set_title("(a) Parks Highway - Median", fontsize='x-large')
ax[0,1].set_title("(b) Parks Highway - Median Low-Noise", fontsize='x-large')
ax[1,0].set_title("(c) Denali Fault - Median", fontsize='x-large')
ax[1,1].set_title("(d) Denali Fault - Median Low-Noise", fontsize='x-large')

ax[0,0].set_ylabel("Power, dB", fontsize='x-large')
ax[1,0].set_ylabel("Power, dB", fontsize='x-large')
ax[1,0].set_xlabel("Frequency, Hz", fontsize='x-large')
ax[1,1].set_xlabel("Frequency, Hz", fontsize='x-large')
#ax[0,1].legend()
#ax[1,1].legend()
for a in ax.flatten():
    a.set_xlim(5.2556e-3, 2.4355e2)
    a.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()
