import obspy
import pyproj
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from obspy.taup import TauPyModel
from obspy.geodetics import kilometer2degrees

folder = '/scratch/naalexeev/NODAL/'
utm_proj = pyproj.Proj(proj='utm', zone='6', ellps='WGS84')
model = TauPyModel(model='iasp91')

# Load the seismometer location data
seismo_data = pd.read_csv('/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/parkshwy_nodes.txt', sep="|")
seismo_latitudes = seismo_data['Latitude']
seismo_longitudes = seismo_data['Longitude']
stations = seismo_data['Station']
bad_data = [1266,1265,1267,1270,1261,1262,1269,1266,1258,1259,1263,1260,1264,1165,1163,1162,1211,1180,1145,1146,1291,1172,1171,1140,1151,1231,1232,1233,1227,1220,1221,1222,1217,1219,1223,1229,1224,1225,1235,1230,1228,1226,1207,1206,1209,1234,1200,1153,1152,1286]

seismo_utm_x, seismo_utm_y = zip(*[utm_proj(lon, lat) for lat, lon in zip(seismo_latitudes, seismo_longitudes)])
dist_min = np.inf
for blast_num in [1,2,3]:
    if blast_num == 1:
        blast_lon_AEC = -148.66
        blast_lat_AEC = 63.98
        blast_lat = blast_lat_AEC
        blast_lon = blast_lon_AEC
        blast_lat = 63.9901
        blast_lon = -148.7392
        depth_AEC = 0
        depth = depth_AEC
    elif blast_num == 2:
        blast_lon_AEC = -148.68
        blast_lat_AEC = 63.97
        blast_lat = blast_lat_AEC
        blast_lon = blast_lon_AEC
        blast_lat = 63.9901
        blast_lon = -148.7392
        depth_AEC = 0
        depth = depth_AEC
    elif blast_num == 3:
        blast_lon_AEC = -148.765
        blast_lat_AEC = 64.010
        blast_lat = 63.9901
        blast_lon = -148.7392
        depth_AEC = 5.51
        depth = 0

    S_wave_dict = {}
    P_wave_dict = {}
    #convert blast location to UTM
    blast_utm_x, blast_utm_y = utm_proj(blast_lon, blast_lat)
    fig, ax = plt.subplots(figsize=(10, 12))
    for i, station in enumerate(stations):
        if int(station) in bad_data:
            continue
        #find distance between station and blast location
        dist_km = np.sqrt((np.sqrt((seismo_utm_x[i]-blast_utm_x)**2 +(seismo_utm_y[i]-blast_utm_y)**2)/1000)**2 + depth**2)
        if dist_km < dist_min:
            dist_min = dist_km
        if blast_num == 1:
            file = folder + f'2019-02-16T01:00:00.000000Z.2019-02-16T02:00:00.000000Z.{station}.mseed'
        elif blast_num == 2:
            file = folder + f'2019-02-23T02:00:00.000000Z.2019-02-23T03:00:00.000000Z.{station}.mseed'
        elif blast_num == 3:
            file = folder + f'2019-03-02T01:00:00.000000Z.2019-03-02T02:00:00.000000Z.{station}.mseed'
        if Path(file).exists():
            tr = obspy.read(file)
            if blast_num == 1:
                tr[2].trim(tr[2].stats.starttime + (52 * 60) + 21 ,tr[2].stats.starttime + (52 * 60) + 21 + 300)
            elif blast_num == 2:
                tr[2].trim(tr[2].stats.starttime + (11 * 60) + 12 ,tr[2].stats.starttime + (11 * 60) + 12 + 300)
            elif blast_num == 3:
                tr[2].trim(tr[2].stats.starttime + (42 * 60) + 19 ,tr[2].stats.starttime + (42 * 60) + 19 + 250)
            tr[2].filter('bandpass', freqmin=1, freqmax=4, corners=2, zerophase=True)
            data = tr[2][:]
            t_wf = tr[2].times()
        else:
            continue
        arrivals_P = model.get_travel_times(source_depth_in_km=depth,distance_in_degree = kilometer2degrees(dist_km),phase_list=["P"])
        arrivals_S = model.get_travel_times(source_depth_in_km=depth,distance_in_degree = kilometer2degrees(dist_km),phase_list=["S"])

        try:
            go = arrivals_P[0].time
            go = arrivals_S[0].time
        except:
            continue

        P_wave_dict[dist_km] = arrivals_P[0].time
        S_wave_dict[dist_km] = arrivals_S[0].time

        norm_data = (data / np.max(np.abs(data))) * 1.5

        if seismo_latitudes[i] >= blast_lat:
            continue
            ax.plot(t_wf, norm_data + dist_km, 'b', linewidth=0.5)
        elif seismo_latitudes[i] < blast_lat:
            ax.plot(t_wf, norm_data + dist_km, 'k', linewidth=0.5, zorder=10)

    ax.plot(P_wave_dict.values(), P_wave_dict.keys(), color='red', linestyle='-', linewidth=1, zorder=0)
    ax.plot(S_wave_dict.values(), S_wave_dict.keys(), color='orange', linestyle='dashdot', linewidth=1, zorder=0)

    x_top = [57,195]
    y_top = [18.7,64.5]
    # Fit a simple linear regression
    m, b = np.polyfit(x_top, y_top, 1)


    # Extend the line infinitely using one point and the calculated slope
    # The line will go across the entire plot area
    ax.axline(xy1=(x_top[0], y_top[0]), slope=m, color='b', linestyle='--', linewidth=1, alpha=0.8, zorder=0)

    #ax.plot(x_top, y_top, color='b', linestyle='--', linewidth=1, alpha=0.8, zorder=0)
    if blast_num == 1:
        ax.set_xlabel('Time after 2019-02-16 01:52:21 UTC, s', fontsize='x-large')
    elif blast_num == 2:
        ax.set_xlabel('Time after 2019-02-23 02:11:11 UTC, s', fontsize='x-large')
    elif blast_num == 3:
        ax.set_xlabel('Time after 2019-03-02T01:42:10 UTC, s', fontsize='x-large')
    ax.set_ylabel('Southern distance from source, km', fontsize='x-large')
    ax.set_ylim(dist_min-1,200)
    ax.set_xlim(0, 220)
    plt.tight_layout()

    plt.savefig(f'blast_{blast_num}_record_section.png')
    plt.close()