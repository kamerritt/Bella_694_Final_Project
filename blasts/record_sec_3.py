import obspy
import pyproj
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from obspy.taup import TauPyModel
from obspy.geodetics import kilometer2degrees
from matplotlib.ticker import FuncFormatter

model = TauPyModel(model='iasp91')
folder = '/scratch/naalexeev/NODAL/'
normalize_trace_individually = False
utm_proj = pyproj.Proj(proj='utm', zone='6', ellps='WGS84')

# Load the seismometer location data
seismo_data = pd.read_csv('/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/parkshwy_nodes.txt', sep="|")
seismo_latitudes = seismo_data['Latitude']
seismo_longitudes = seismo_data['Longitude']
stations = seismo_data['Station']
bad_data = [1266,1265,1267,1270,1261,1262,1269,1266,1258,1259,1263,1260,1264,1165,1163,1162,1211,1180,1145,1146,1291,1172,1171,1140,1151,1232,1233,1227,1220,1221,1222,1217,1219,1223,1229,1224,1225,1235,1230,1228,1226,1207,1206,1209,1234,1200,1153,1152,1286,1218,1216]
seismo_utm_x, seismo_utm_y = zip(*[utm_proj(lon, lat) for lat, lon in zip(seismo_latitudes, seismo_longitudes)])

blast_lon_AEC = -148.765
blast_lat_AEC = 64.010
blast_lat = 63.9901
blast_lon = -148.7392
depth_AEC = 5.51
depth = 0
road_lat = 63.9901
road_lon =  -149.128404464516

#convert blast location to UTM
blast_utm_x, blast_utm_y = utm_proj(blast_lon, blast_lat)
road_utm_x, road_utm_y = utm_proj(road_lon, road_lat)
dist_2_road =  np.sqrt((blast_utm_x - road_utm_x)**2 + (blast_utm_y - road_utm_y)**2) / 1000  

S_wave_dict = {}
P_wave_dict = {}
fig, axs = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[(65-17.8)/(74-17), 1], sharex=True)
for i, station in enumerate(stations):
    if int(station) in bad_data:
        continue

    #find distance between station and blast location
    dist_km = np.sqrt((seismo_utm_x[i]-blast_utm_x)**2 +(seismo_utm_y[i]-blast_utm_y)**2)/1000
    file = folder + f'2019-03-02T01:00:00.000000Z.2019-03-02T02:00:00.000000Z.{station}.mseed'

    if Path(file).exists():
        tr = obspy.read(file)
        tr[2].trim(tr[2].stats.starttime + (42 * 60) + 19 ,tr[2].stats.starttime + (42 * 60) + 19 + 250)
        tr[2].filter('bandpass', freqmin=1, freqmax=4, corners=2, zerophase=True)
        start = tr[2].stats.starttime
        data = tr[2][:]
        t_wf = tr[2].times()
    else:
        continue

    if normalize_trace_individually:
        norm_data = data / np.max(np.abs(data))
    else:
        norm_data = data / 500
    
    if seismo_latitudes[i] >= blast_lat:
        axs[0].plot(t_wf, norm_data + dist_km, 'k', linewidth=0.5)
        tt = 0
        dist_km = dist_km
    elif seismo_latitudes[i] < blast_lat:
        axs[1].plot(t_wf, norm_data - dist_km, 'k', linewidth=0.5)
        #axs[1].text(-5, -dist_km, str(station))
        tt = 1
        dist_km = -dist_km
        if int(station) in [1231]:
            axs[0].plot(t_wf, norm_data - dist_km, 'k', linewidth=0.5)
            dist_fake_station = -dist_km
    arrivals_P = model.get_travel_times(source_depth_in_km=0,distance_in_degree = kilometer2degrees(dist_km),phase_list=["P"])
    arrivals_S = model.get_travel_times(source_depth_in_km=0,distance_in_degree = kilometer2degrees(dist_km),phase_list=["S"]) 

    try:
        go = arrivals_P[0].time
        go = arrivals_S[0].time
    except:
        continue

    P_wave_dict[dist_km] = arrivals_P[0].time
    S_wave_dict[dist_km] = arrivals_S[0].time

    if int(station) in [1231]:
        P_wave_dict[-dist_km] = arrivals_P[0].time
        S_wave_dict[-dist_km] = arrivals_S[0].time

for tt in [0,1]:
    axs[tt].plot(P_wave_dict.values(), P_wave_dict.keys(), color='red', linestyle='-', linewidth=1, zorder=0)
    axs[tt].plot(S_wave_dict.values(), S_wave_dict.keys(), color='orange', linestyle='dashdot', linewidth=1, zorder=0)
x_top = [57,195]
y_top = [18.7,64.5]
slope_top = (y_top[1]-y_top[0])/(x_top[1]-x_top[0])

x_bottom = [57, 202]
y_bottom = [-18.7, -66.1]
slope_bottom = (y_bottom[1]-y_bottom[0])/(x_bottom[1]-x_bottom[0])

axs[0].plot(x_top, y_top, color='b', linestyle='--', linewidth=1, alpha=0.8, zorder=0)
axs[1].plot(x_bottom, y_bottom, color='b', linestyle='--', linewidth=1, alpha=0.8, zorder=0)

axs[0].text(215, 62, r"$c_{eff}$" + f" =  {slope_top*1000:.0f} m/s", fontsize=12, color='b',
            bbox=dict(facecolor='white', alpha=0.95, edgecolor='none', pad=2))
axs[1].text(215, -67, r"$c_{eff}$" + f" =  {abs(slope_bottom)*1000:.0f} m/s", fontsize=12, color='b',
            bbox=dict(facecolor='white', alpha=0.95, edgecolor='none', pad=2))

plt.tight_layout()
plt.subplots_adjust(hspace=0, bottom=0.05)

if normalize_trace_individually:
    axs[0].set_ylim(18,65)
    axs[1].set_ylim(-68,-18)
else:
    axs[0].set_ylim(dist_fake_station,65)
    axs[1].set_ylim(-68,-dist_fake_station)

# Format bottom axis tick labels to show positive numbers (absolute values)
def abs_formatter(val, pos):
    # show integers without decimal when possible
    if abs(val - round(val)) < 1e-6:
        return f"{int(abs(round(val)))}"
    return f"{abs(val):.1f}"

axs[1].yaxis.set_major_formatter(FuncFormatter(abs_formatter))

axs[0].spines['bottom'].set_visible(False)
axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axs[1].spines['top'].set_visible(True)
axs[1].spines['top'].set_linewidth(0.1)
# Label showing seconds after the plot start time (UTC)
plot_start_utc = obspy.UTCDateTime("2019-03-02T01:00:00.000000Z") + 42*60 + 19
plt.xlabel(f"Time after {plot_start_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}, s", fontsize='x-large')
axs[0].set_ylabel('Northern distance from source, km', fontsize='x-large')
axs[1].set_ylabel('Southern distance from source, km', fontsize='x-large')
plt.xlim(0, 250)
plt.show()