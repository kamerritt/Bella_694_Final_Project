import pandas as pd
import matplotlib.pyplot as plt

from soundscape_psd_functions import parse_xml_file, parse_psds_grouped_by_day, compute_daily_medians, median_data

# Function to plot frequencies vs powers
def plot_data(frequencies, powers, color='black'):
    plt.plot(frequencies, powers, color=color, alpha=0.5)
    plt.title('Frequency vs Power: Node '+ station)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.xscale('log')
    plt.xlim(-2,250)

seismo_data = pd.read_csv('/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/parkshwy_nodes.txt', sep="|")
stations = seismo_data['Station']

for station in stations:
    station = str(station)
    sta, freqs, powers = parse_xml_file(station,channel="Z")
    med_freq, med_power = median_data(freqs, powers)
    xml_file = f"psd_stations/psd_{station}_DPZ.xml"

    for day, (freqs, powers) in compute_daily_medians(parse_psds_grouped_by_day(xml_file)).items():
        plot_data(freqs, powers)
        plot_data(med_freq, med_power, color='red')

    plt.savefig("psd_"+station+".png")
    plt.close()
