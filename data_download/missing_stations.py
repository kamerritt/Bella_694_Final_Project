import pandas as pd
import numpy as np

nodes = STATIONS_txt = "/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/full_nodes.txt"
seismo_data = pd.read_csv(nodes, sep="|")

stations = seismo_data['Station'].astype(int).values

station_check = np.arange(1001, 1306)  
missing_stations = [station for station in station_check if station not in stations and station <= 1305]
print("Missing stations between 1001 and 1305:")
print(missing_stations)

station_check = np.arange(1500, 1590)  
missing_stations = [station for station in station_check if station not in stations and station >= 1500]
print("Missing stations between 1500 and 1590:")
print(missing_stations)