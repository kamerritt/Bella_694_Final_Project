import sys
import pyproj
import pandas as pd
import numpy as np

from scipy.signal import spectrogram
from pathlib import Path
from matplotlib import pyplot as plt
from datetime import datetime, timedelta, timezone
# Ensure repository root is on sys.path so local package 'src' can be imported
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.main_inv_fig_functions import remove_median
from src.doppler_funcs import flight_list, time_check, find_closest_point, get_equip, closest_time_calc, load_flight_file, avg_return, load_waveform

def dist_less(flight_utm_x_km, flight_utm_y_km, seismo_utm_x_km, seismo_utm_y_km):
	f = False
	for s in range(len(flight_utm_x_km)):
		for l in range(len(seismo_utm_x_km)):
			dist_km = np.sqrt((seismo_utm_y_km[l]-flight_utm_y_km[s])**2 +(seismo_utm_x_km[l]-flight_utm_x_km[s])**2)
			if dist_km <= 50:
				f = True
				break
			else:
				continue
	return f
def time_check(closest_time, start_time, end_time, s_index):

	v = False
	# Convert the times to datetime objects
	time_airplane = datetime.fromtimestamp(closest_time, tz=timezone.utc)
	start_time_obj = datetime.strptime(start_time[s_index], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
	end_time_obj = datetime.strptime(end_time[s_index], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)

	if time_airplane < start_time_obj or time_airplane > end_time_obj:
		v = True
	return v

# Load flight files and filenames
flight_files,filenames = flight_list(2, 3, 22, 23)

utm_proj = pyproj.Proj(proj='utm', zone='6', ellps='WGS84')

station_is = np.arange(1123,1246)
# Load the seismometer location data
seismo_data = pd.read_csv('/home/irseppi/REPOSITORIES/parkshwynodal/input/parkshwy_nodes.txt', sep="|")
# ensure Station is int and keep only stations 1123-1245 (station_is defined above)
seismo_data['Station'] = seismo_data['Station'].astype(int)
seismo_data = seismo_data[seismo_data['Station'].isin(station_is)].set_index('Station').reindex(station_is).reset_index()

seismo_latitudes = seismo_data['Latitude'].to_numpy()
seismo_longitudes = seismo_data['Longitude'].to_numpy()
sta = seismo_data['Station'].to_numpy()
start_time = seismo_data['StartTime'].to_numpy()
end_time = seismo_data['EndTime'].to_numpy()

for flight_file, filename, in zip(flight_files, filenames):
    # Convert latitude and longitude to UTM coordinates
    seismo_utm = [utm_proj(lon, lat) for lat, lon in zip(seismo_latitudes, seismo_longitudes)]
    seismo_utm_x, seismo_utm_y = zip(*seismo_utm)

    # Convert UTM coordinates to kilometers
    seismo_utm_x_km = [x / 1000 for x in seismo_utm_x]
    seismo_utm_y_km = [y / 1000 for y in seismo_utm_y]

    seismo_utm_km = [(x, y) for x, y in zip(seismo_utm_x_km, seismo_utm_y_km)]
    seismo_utm_x_km, seismo_utm_y_km = zip(*seismo_utm_km)
    flight_utm_x_km, flight_utm_y_km, flight_path, timestamp, alt, speed, head, flight_num, date = load_flight_file(flight_file,filename)

    if int(date) != 20190222:
        continue

    equip = get_equip(date, flight_num)

    # Iterate over seismometer data
    for s in range(len(seismo_data)):
        seismometer = (seismo_utm_x_km[s], seismo_utm_y_km[s])  
        station = sta[s]

        if dist_less(flight_utm_x_km, flight_utm_y_km, seismo_utm_x_km, seismo_utm_y_km) == False:
            continue

        closest_p, d, index= find_closest_point(flight_path, seismometer)
        if index == None:
            continue
        #if d <= 50:

        closest_x, closest_y = closest_p
        closest_time = closest_time_calc(closest_p, flight_path, timestamp, index)

        if time_check(closest_time, start_time, end_time,s) == True:
            continue

        alt_avg_m, speed_avg_mps, head_avg = avg_return(alt, speed, head, index)

        waveform_start_time = closest_time - 600
        #1169
        #529935813
        base_date = datetime.strptime(str(date), '%Y%m%d')
        time_dt = base_date + timedelta(seconds=waveform_start_time)
        print(date, time_dt.hour)
        # keep only events between 19:40 and 20:30 (inclusive) UTC on that date
        start_window = base_date + timedelta(hours=19, minutes=40)
        end_window = base_date + timedelta(hours=20, minutes=30)
        if time_dt < start_window or time_dt > end_window:
            continue
        print(date, time_dt.hour)
        print(station)
        print(flight_num)
        print(d,alt_avg_m, speed_avg_mps, head_avg)
        waveform_start_time = closest_time - 120
        data, fs, t_wf, title = load_waveform(station, waveform_start_time)
        frequencies, times, Sxx = spectrogram(data, fs, scaling='density', nperseg=fs, noverlap=fs * .9, detrend = 'constant')
        if len(times) == 0 or len(frequencies) == 0 or len(Sxx) == 0:
            continue
        
        spec, MDF = remove_median(Sxx)
        middle_index =  len(times) // 2
        middle_column = spec[:, middle_index]
        vmin = 0  
        vmax = np.max(middle_column) 

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(8,6))     
        ax1.plot(t_wf, data, 'k', linewidth=0.5)
        ax1.set_title(title)

        # Plot spectrogram
        cax = ax2.pcolormesh(times, frequencies, spec, shading='gouraud', cmap='pink_r', vmin=vmin, vmax=vmax)		
        plt.show()
        plt.close()


#20190222 21
#1220
#529956620
#16.257464909431658 7307.58 231.242578 215.0