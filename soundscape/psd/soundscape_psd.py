import numpy as np

from matplotlib import pyplot as plt
from soundscape_psd_functions import load_stations_for_distance_calculation, median_data, parse_xml_file, align_and_median_psd, station_low_noise_med_per_freq


find_clear_station = False

seismo_data = load_stations_for_distance_calculation(fullstations=False)
stations = seismo_data['Station']
lat = seismo_data['Latitude']

color = np.zeros((len(stations), 125))
station_freqs = {}
station_powers = {}
x = []
y = []

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
    y.append(station)
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
plt.pcolormesh(x, y, color, vmin=-45, vmax=45, cmap='seismic')

if find_clear_station:
    # draw horizontal lines at the rows for stations 1264 and 1281
    for target in (1260, 1281):
        try:
            idx = list(stations).index(target)              # if stations are ints
        except ValueError:
            try:
                idx = list(map(str, stations)).index(str(target))  # if stations are strings
            except ValueError:
                print(f"Station {target} not found")
                continue
    plt.axhline(y=idx, color='k', linestyle='--', linewidth=1)
# show every 10th station starting at the first (use actual station names)
tick_positions = np.arange(0, len(stations), 1)
tick_labels = []
first_station = stations.iloc[0]
last_station = stations.iloc[-1]
for p in tick_positions:
    if p == first_station:
        tick_labels.append(str(first_station))
    elif p == last_station:
        tick_labels.append(str(last_station))
    elif str(stations[p]) == "1304":
        tick_labels.append("Nenana")
    elif str(stations[p]) == "1256":
        tick_labels.append("NFFTB")
        plt.axhline(y=p-1, color='k', linestyle='--', linewidth=1, zorder=1000)
    elif str(stations[p]) == "1215":
        tick_labels.append("Healy")
    elif str(stations[p]) == "1199":
        tick_labels.append("D. Park Rd")
    elif str(stations[p]) == "1163":
        tick_labels.append("Denali Fault")
        plt.axhline(y=p-1, color='k', linestyle='--', linewidth=1, zorder=1000)
    elif str(stations[p]) == "1153":
        tick_labels.append("Cantwell")
    elif str(stations[p]) == "1003":
        tick_labels.append("Trapper Creek")
    elif str(stations[p]) == "1278":
        tick_labels.append("Anderson")
    elif str(stations[p]) == "1022": #1049 #1029
        tick_labels.append("Susitna Basin")
        plt.axhline(y=p, color='k', linestyle='--', linewidth=1, zorder=1000)
    elif str(stations[p]) == "1219":
        tick_labels.append("HCF") #Healy Creek fault
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
plt.axvline(x=0.1, color='k', linestyle='--', linewidth=1)
plt.axvline(x=1, color='k', linestyle='--', linewidth=1)
plt.axvline(x=10, color='k', linestyle='--', linewidth=1)
plt.axvline(x=100, color='k', linestyle='--', linewidth=1)
plt.yticks(tick_positions, tick_labels)
   
plt.xscale('log')
plt.xlabel('Frequency, Hz', fontsize='x-large')
cbar = plt.colorbar(label='Power Difference, dB', fraction=0.02, pad=0.01, aspect=50)

plt.show()
#plt.savefig('soundscape.pdf')

