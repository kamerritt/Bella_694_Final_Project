import numpy as np
import pandas as pd

from soundscape_psd_functions import parse_xml_file, median_data

STATIONS_txt = "/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/full_nodes.txt"
stations_df = pd.read_csv(STATIONS_txt, sep="|")

stations = stations_df['Station'].tolist()
latitudes = stations_df['Latitude'].tolist()
longitudes = stations_df['Longitude'].tolist()
elevations = stations_df['Elevation'].tolist()

channels = ['1','2','Z']

OUTPUT = open('db_noise.txt', 'w')
OUTPUT.write('network,station,channel,longitude,latitude,elevation,power_value,\n')
for i, station in enumerate(stations):
    for channel in channels:
        station, frequencies, powers = parse_xml_file(station, channel)
        f, p = median_data(frequencies, powers)
        power_med_norm = np.sqrt(np.sum(np.array(p)**2))
        OUTPUT.write(f'ZE,{station},DP{channel},{longitudes[i]},{latitudes[i]},{elevations[i]},{power_med_norm},\n')
OUTPUT.close()