import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from matplotlib import pyplot as plt
from soundscape_psd_functions import parse_xml_file

def station_low_noise_median_artificial(frequencies, powers, n_low=10):
    psd_by_freq = {}
    for i, freq in enumerate(frequencies):
        if freq not in psd_by_freq:
            psd_by_freq[freq] = []
        psd_by_freq[freq].append(powers[i])

    med_psd_by_freq = []
    for f in psd_by_freq.keys():
        # Looking for indices for the 3 largest difference values
        lowest_freqs = np.sort(psd_by_freq[f])[0:n_low]
        med_psd_by_freq.append(np.median(lowest_freqs))

    return med_psd_by_freq 


def station_low_noise_median(station,channel="Z", n_low=10):
    """
    Return the median PSD (per frequency) of the
    10 lowest-norm 30-minute PSDs in an XML file.
    """
    xml_file = f"psd_stations/psd_{station}_DP{channel}.xml"

    tree = ET.parse(xml_file)
    root = tree.getroot()

    psds = root.findall(".//Psd")

    spectra = []
    norms = []
    freqs = None

    for psd in psds:
        values = psd.findall("value")

        f = np.array([float(v.attrib["freq"]) for v in values])
        p_db = np.array([float(v.attrib["power"]) for v in values])

        if freqs is None:
            freqs = f

        # Use linear power ONLY for ranking
        p_lin = 10 ** (p_db / 10.0)

        # Norm of this 30-min PSD
        norm = np.linalg.norm(p_lin)

        spectra.append(p_db)
        norms.append(norm)

    spectra = np.array(spectra)   
    norms = np.array(norms)

    # Indices of the 10 quietest PSDs
    low_idx = np.argsort(norms)[:n_low]

    # Median power at EACH frequency across those 10 PSDs
    low_noise_median = np.median(spectra[low_idx, :], axis=0)

    return low_noise_median

def median_data(frequencies, powers):
    freq_dict = {}
    for i, freq in enumerate(frequencies):
        if freq not in freq_dict:
            freq_dict[freq] = []
        freq_dict[freq].append(powers[i])
    med_powers = []
    for f in freq_dict.keys():
        med_powers.append(np.median(freq_dict[f]))
    return list(freq_dict.keys()), med_powers

seismo_data = pd.read_csv('/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/parkshwy_nodes.txt', sep="|")
stations = seismo_data['Station']

color = np.zeros((len(stations), 125))

x = []
y = []

for i, station in enumerate(stations):
    station = str(station)
    _, frequencies, powers = parse_xml_file(station,channel="Z")

    freqs, med_powers = median_data(frequencies, powers)
    low_noise_med = station_low_noise_median_artificial(freqs, powers, n_low=10)
    #low_noise_med = station_low_noise_median(station,channel="Z", n_low=10)

    y.append(station)
    if i == 0:
        x.append(freqs)
    for j in range(len(med_powers)):
        color[i,j] = med_powers[j] - low_noise_med[j]

#take the median value for every column in the color matrix
color_med = np.median(color, axis=0)
for tt in range(len(stations)):
    color[tt,:] = color[tt,:] - color_med

plt.figure()
# set 0 to be white
max_abs = np.max(np.abs(color))
plt.pcolormesh(x, y, color, cmap = 'seismic', vmin=-max_abs, vmax=max_abs)

# show every 10th station starting at the first (use actual station names)
tick_positions = np.arange(0, len(stations), 5)
tick_labels = [str(stations[p]) for p in tick_positions]
plt.yticks(tick_positions, tick_labels)
   
plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Station')
plt.colorbar(label='Median power')
plt.show()