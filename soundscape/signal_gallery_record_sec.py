import obspy
import numpy as np
import matplotlib.dates as mdates

from pathlib import Path
from matplotlib import pyplot as plt
from scipy.signal import spectrogram
from datetime import timezone

folder = '/scratch/irseppi/500sps/2019_02_22/'

stations = np.arange(1123, 1246)
fig, ax = plt.subplots(2, 1, figsize=(12, 8),sharex=True,gridspec_kw={'height_ratios': [1, 5]})
for i,station in enumerate(stations):

    file = folder + f'ZE_{station}_DPZ.msd'
    if Path(file).exists():
        tr = obspy.read(file)
        tr[0].trim(tr[0].stats.starttime + (19 * 60 * 60) +(40 * 60)  ,tr[0].stats.starttime + (19 * 60 * 60) + (140 * 60))
        
        data = tr[0][:]
        t_wf = tr[0].times()
        norm_data = (data / np.max(np.abs(data)))
        if i == 0:
            # convert spectrogram times (seconds since trace start) to timezone-aware datetimes and then to matplotlib date numbers
            start_utc = tr[0].stats.starttime
            times_abs = [(start_utc + t).datetime.replace(tzinfo=timezone.utc) for t in t_wf]
            time_nums = mdates.date2num(times_abs)
        ax[1].plot(time_nums, norm_data + int(station), 'k', linewidth=0.5)

frequencies, times, Sxx = spectrogram(data, 500, scaling='density', nperseg=500, noverlap=int(500 * .6), detrend='constant')

a, b = Sxx.shape
MDF = np.zeros((a,b))
for row in range(len(Sxx)):
    median = np.median(Sxx[row])
    MDF[row, :] = median

# Avoid log10(0) by replacing zeros with a small positive value
Sxx_safe = np.where(Sxx == 0, 1e-10, Sxx)
MDF_safe = np.where(MDF == 0, 1e-10, MDF)

spec = 10 * np.log10(Sxx_safe) - (10 * np.log10(MDF_safe))
max_val = np.max(spec)*0.8
min_val = np.min(spec)*0.1

# convert spectrogram times (seconds since trace start) to timezone-aware datetimes and then to matplotlib date numbers
times_abs = [(start_utc + t).datetime.replace(tzinfo=timezone.utc) for t in times]
time_nums = mdates.date2num(times_abs)

# plot using matplotlib date numbers on the x axis
pcm = ax[0].pcolormesh(time_nums, frequencies, spec, cmap='pink_r', vmin=min_val, vmax=max_val)
ax[1].set_xlim(time_nums.min(), time_nums.max())

air = [mdates.date2num(start_utc.datetime.replace(hour=19, minute=48, second=15, tzinfo=timezone.utc)),
      mdates.date2num(start_utc.datetime.replace(hour=19, minute=58, second=50, tzinfo=timezone.utc)), 
      mdates.date2num(start_utc.datetime.replace(hour=20, minute=3, second=15, tzinfo=timezone.utc)), 
      mdates.date2num(start_utc.datetime.replace(hour=20, minute=19, second=00, tzinfo=timezone.utc))]
ax[1].plot(air, [1245, 1209, 1193, 1123], 'r', linewidth=0.7)
ax[0].axvline(mdates.date2num(start_utc.datetime.replace(hour=19, minute=48, second=15, tzinfo=timezone.utc)), linestyle='--', linewidth=0.7, color='r', alpha=0.7)

eq = [mdates.date2num(start_utc.datetime.replace(hour=20, minute=37, second=21, tzinfo=timezone.utc)),
      mdates.date2num(start_utc.datetime.replace(hour=20, minute=38, second=0, tzinfo=timezone.utc))]
ax[1].plot(eq, [1245,  1130], c='blue', linewidth=0.7)
ax[0].axvline(mdates.date2num(start_utc.datetime.replace(hour=20, minute=37, second=21, tzinfo=timezone.utc)), linestyle='--',linewidth=0.7, color='blue', alpha=0.7)

train = [mdates.date2num(start_utc.datetime.replace(hour=20, minute=58, second=00, tzinfo=timezone.utc)),
      mdates.date2num(start_utc.datetime.replace(hour=21, minute=9, second=0, tzinfo=timezone.utc))]
ax[1].plot(train, [1245,  1228], c='magenta',  linewidth=0.7)
ax[0].axvline(mdates.date2num(start_utc.datetime.replace(hour=20, minute=58, second=00, tzinfo=timezone.utc)), linestyle='--', linewidth=0.7, color='magenta', alpha=0.7)

car = [mdates.date2num(start_utc.datetime.replace(hour=20, minute=8, second=35, tzinfo=timezone.utc)),
      mdates.date2num(start_utc.datetime.replace(hour=21, minute=26, second=15, tzinfo=timezone.utc))]
ax[1].plot(car, [1245,  1123], c='purple', linewidth=0.7)
ax[0].axvline(mdates.date2num(start_utc.datetime.replace(hour=20, minute=8, second=35, tzinfo=timezone.utc)), linestyle='--', linewidth=0.7, color='purple', alpha=0.7)

# format x axis as HH:MM:SS in UTC
ax[1].xaxis_date()
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
ax[1].set_ylim(1122, 1247)
ax[0].set_ylabel("Frequency, Hz", fontsize='large')
ax[0].tick_params(axis='y', labelsize='large')
ax[1].set_xlabel("Time (UTC) — 2019-02-22, HH:MM", fontsize='x-large')
# show every 10th station starting at the first (use actual station names)
tick_positions = np.arange(1122, 1247, 1)
tick_labels = []
first_station = stations[0]
last_station = stations[-1]
for p in tick_positions:
    if p == first_station:
        tick_labels.append(str(first_station))
    elif p == last_station:
        tick_labels.append(str(last_station))
    elif p == 1235:
        tick_labels.append("Ferry")
    elif p == 1215:
        tick_labels.append("Healy")
    elif p == 1196:
        tick_labels.append("Park Road")
    elif p == 1199:
        tick_labels.append("Denali")
    elif p == 1160:
        tick_labels.append("Fault")
    elif p == 1163:
        tick_labels.append("Denali")
    elif p == 1153:
        tick_labels.append("Cantwell")
    else:
        tick_labels.append("")
# Remove tick marks that don't have an associated label (empty string)
filtered = [(p, l) for p, l in zip(tick_positions, tick_labels) if l != ""]
if filtered:
    tick_positions, tick_labels = zip(*filtered)
    tick_positions = np.array(tick_positions)
    tick_labels = list(tick_labels)
else:
    tick_positions = np.array([])
    tick_labels = []

ax[1].set_yticks(tick_positions, tick_labels, fontsize='large')
ax[1].tick_params(axis='x', labelsize='large')

#set the width between plots smaller
plt.subplots_adjust(hspace=0.05)
plt.tight_layout(pad=0, h_pad=0)
plt.savefig('signal_gallery_record_section.png', dpi=400)

start, end = [0,1700,4170], [900,1780,5270]
labels = ['Aircraft', 'Car', 'Train']
steps = [200,10,200]
for s, e, l, step in zip(start, end, labels, steps):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # plot spectrogram for 0-800 s and scale colorbar to max in that range
    mask = (times >= s) & (times <= e)
    vmin_range = np.min(spec[:, mask]) if np.any(mask) else np.min(spec)
    vmax_range = np.max(spec[:, mask]) if np.any(mask) else np.max(spec)
    pcm = ax.pcolormesh(times, frequencies, spec, cmap='pink_r', vmin=min_val, vmax=max_val) #*0.8)
    plt.xlim(s, e)
    ticks = np.arange(s, e + step, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(t - s)) for t in ticks])

    ax.tick_params(axis='both', labelsize='large')

    plt.xlabel("Time, s", fontsize='large')
    plt.ylabel("Frequency, Hz", fontsize='large')
    plt.tight_layout()
    plt.savefig(f'spectrogram_{l.lower()}.png', dpi=300)

