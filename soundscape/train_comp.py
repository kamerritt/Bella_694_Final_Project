import obspy

from pathlib import Path
from matplotlib import pyplot as plt
folder = '/scratch/irseppi/500sps/2019_02_16/'


fig, ax = plt.subplots(1, 1, figsize=(12, 8))

file = folder + f'ZE_1127_DPZ.msd'
if Path(file).exists():

    tr = obspy.read(file)
    tr[0].trim(tr[0].stats.starttime + (16 * 60 * 60) - 424.5 + 200,tr[0].stats.starttime + (16 * 60 * 60) + (200)-100)
    data = tr[0][:]
    t_wf = tr[0].times()
    print(f'{tr[0].stats.network}.{tr[0].stats.station}.{tr[0].stats.location}.{tr[0].stats.channel} − starting {tr[0].stats["starttime"]}')

    ax.plot(t_wf, data, 'k', linewidth=0.5, label='Northbound')

    tr = obspy.read(file)
    tr[0].trim(tr[0].stats.starttime + (14 * 60 * 60) +(15 * 60) + 200,tr[0].stats.starttime + (14 * 60 * 60) + (15 * 60) + (600)-125)  
    data = tr[0][:]
    t_wf = tr[0].times()
    print(f'{tr[0].stats.network}.{tr[0].stats.station}.{tr[0].stats.location}.{tr[0].stats.channel} − starting {tr[0].stats["starttime"]}')
    ax.plot(t_wf, data, 'r', linewidth=0.5, label='Southbound')

ax.set_xlim(0,250)
ax.set_xlabel("Time, s", fontsize='large')
ax.set_ylabel("Amplitude, counts", fontsize='x-large')
ax.legend(loc='upper left', frameon=True, fontsize='x-large')
plt.grid()
plt.show()