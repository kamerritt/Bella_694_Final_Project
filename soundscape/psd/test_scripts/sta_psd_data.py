import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from soundscape_psd_functions import parse_xml_file, median_data, parse_psds_grouped_by_day, compute_daily_medians

# Function to plot frequencies vs powers
def plot_data(frequencies, powers, color='black'):
    plt.plot(frequencies, powers, color=color, alpha=0.5)
    plt.title('Frequency vs Power: Node '+ station)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power [10log10(m**2/sec**4/Hz)](dB)')
    plt.xscale('log')
    plt.xlim(5.2556*10**-3,2.4355*10**2)

# Main function to process the XML file
def process_xml(xml_file):
    sta, frequencies, powers = parse_xml_file(station,channel="Z")
    med_freq, med_power = median_data(frequencies, powers)
    for day, (freqs, powers) in compute_daily_medians(parse_psds_grouped_by_day(xml_file)).items():
        plot_data(freqs, powers)
        plot_data(med_freq, med_power, color='red')
    plt.grid()
    plt.show()
    #plt.savefig("/scratch/irseppi/nodal_data/psd_noise/psd_"+station+".png")
    plt.close()

seismo_data = pd.read_csv('/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/parkshwy_nodes.txt', sep="|")
stations = seismo_data['Station']

for station in stations:
    station = str(station)
    xml_file = "psd_stations/psd_"+station+".xml"

    process_xml(xml_file)

#################################################################################################################################################

#Example to plot as MUSTANG plots PSDs
plot_MUSTANG_style = False
if plot_MUSTANG_style:

    station = '1030' # Example station
    xml_file = "psd_"+station+".xml"

    tree = ET.parse(xml_file)
    root = tree.getroot()
    plt.figure()
    # Loop through all 'Psd' elements
    for psd in root.findall(".//Psd"):
        frequencies = []
        powers = []
        # Loop through all 'value' elements within each 'Psd'
        for value in psd.findall(".//value"):
            freq = float(value.get("freq"))
            power = float(value.get("power"))
            frequencies.append(freq)
            powers.append(power)
        plt.plot(frequencies, powers)
    plt.xscale('log')
    plt.xlim(2.4355*10**2, 5.2556*10**-3)
    plt.grid()
    plt.savefig("MUSTANG_psd_"+station+".png")