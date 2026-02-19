import obspy
import pyproj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from obspy.signal.trigger import classic_sta_lta
from obspy.taup import TauPyModel
from obspy.geodetics import kilometer2degrees

utm_proj = pyproj.Proj(proj='utm', zone='6', ellps='WGS84')
model = TauPyModel(model="ak135")

# Load the seismometer location data
seismo_data = pd.read_csv('/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/parkshwy_nodes.txt', sep="|")
seismo_latitudes = seismo_data['Latitude']
seismo_longitudes = seismo_data['Longitude']
stations = seismo_data['Station']
seismo_utm_x, seismo_utm_y = zip(*[utm_proj(lon, lat) for lat, lon in zip(seismo_latitudes, seismo_longitudes)])

otime = "2019-02-18T17:02:46.332Z"
olon = -149.93
olat = 61.46
event_depth = 40.91 #km
mag = 4.36

#convert eq location to UTM
outm_x, outm_y = utm_proj(olon, olat)

# Set plot limits
minoffset = 95 # kmeters
maxoffset = 350 # kmeters
timeStart = 0
timeLength = 300

folder = '/scratch/irseppi/500sps/2019_02_18/'

vmodel = True
wave_dict = {}

#100 Hz
#Instrament response removal
plt.figure(figsize=(10, 6))
for i,station in enumerate(stations):
    file = folder + f'ZE_{station}_DPZ.msd'

    if Path(file).exists():
        dist_km = np.sqrt((seismo_utm_x[i] - outm_x)**2 +(seismo_utm_y[i]-outm_y)**2)/1000
        tr = obspy.read(file)
        tr[0].trim(tr[0].stats.starttime + (17 * 60 * 60) + (2 * 60) + 46.332 ,tr[0].stats.starttime  + (17 * 60 * 60) + (2 * 60) + 46.332 + 300)
        #filter bandpass 3- 8 Hz
        #tr[0].filter("bandpass", freqmin=3, freqmax=8, corners=4)
        #tr[0].filter("highpass", freq=1, corners=4)
        data = tr[0][:]
        t_wf = tr[0].times()
        df = tr[0].stats.sampling_rate
        num_samples = tr[0].stats.npts

        sta = int(1 * df)
        lta = int(5 * df)

        sta_lta = classic_sta_lta(tr[0].data, sta, lta)
        dist_x = np.full(num_samples, dist_km)
        timelist = np.arange(num_samples) / df


        plt.scatter(
            dist_x, timelist,
            c=sta_lta,
            s=0.7,
            cmap="Spectral_r",
            vmin=0, vmax=5,
            rasterized=True
        )
        if vmodel:
            arrival_s = model.get_travel_times(source_depth_in_km=event_depth,
                        distance_in_degree=kilometer2degrees(dist_km))

            if arrival_s:
                for tt in range(len(arrival_s)):
                    if dist_km not in wave_dict:
                        wave_dict[dist_km] = []
                    wave_dict[dist_km].append(arrival_s[tt].time)

if vmodel:
    for dist, arrival_times in wave_dict.items():
        for arrival_time in arrival_times:
            plt.scatter([dist], [arrival_time], color='k', s=2, zorder=1000)

plt.xlabel("Distance from epicenter, km")
plt.ylabel("Time, s")
plt.xlim(minoffset, maxoffset)
plt.ylim(timeStart + 5, timeStart + 110)
plt.gca().invert_yaxis()
plt.title(f"otime: {otime} lon: {olon} lat: {olat} depth: {event_depth} km mag: {mag}")
plt.colorbar(label="STA/LTA")
plt.tight_layout()
#plt.legend(loc="lower left")
plt.savefig(f'denali_M4.3event_DPZ_staLta.png', dpi=300)

