import obspy 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

tr = obspy.read("/scratch/naalexeev/NODAL/2019-03-02T01:00:00.000000Z.2019-03-02T02:00:00.000000Z.1248.mseed") #1224,1226 clean 
#1249 very intresting local signal!

# Compute spectrogram
data = tr[2][:]
fs = int(tr[2].stats.sampling_rate)
frequencies_lta, times_lta, Sxx_lta = spectrogram(data, fs, scaling='spectrum', nperseg=fs, noverlap=fs * .99, detrend = 'constant') 

tr[2].trim(tr[2].stats.starttime + 42 * 60 + 10 , tr[2].stats.starttime + 43 * 60 + 80)
data = tr[2][:]
fs = int(tr[2].stats.sampling_rate)
title = f'{tr[2].stats.network}.{tr[2].stats.station}.{tr[2].stats.location}.{tr[2].stats.channel} − starting {tr[2].stats["starttime"]}'                        
t_wf = tr[2].times()

# Compute spectrogram
frequencies, times, Sxx = spectrogram(data, fs, scaling='spectrum', nperseg=fs, noverlap=fs * .99, detrend = 'constant') 

a, b = Sxx.shape

MDF = np.zeros((a,b))
for row in range(len(Sxx)):
    median = np.median(Sxx_lta[row])
    MDF[row, :] = median

# Avoid log10(0) by replacing zeros with a small positive value
Sxx_safe = np.where(Sxx == 0, 1e-10, Sxx)
MDF_safe = np.where(MDF == 0, 1e-10, MDF)

spec = 10 * np.log10(Sxx_safe) #- (10 * np.log10(MDF_safe))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,6))     

ax1.plot(t_wf, data, 'k', linewidth=0.5)
ax1.set_title(title)
ax1.set_ylabel('Amplitude, Counts')
ax1.margins(x=0)
ax1.set_position([0.125, 0.6, 0.775, 0.3]) 

# Plot spectrogram
cax = ax2.pcolormesh(times, frequencies, spec, cmap='plasma_r', vmin = 0, vmax = np.max(spec)) 
ax2.set_xlabel('Time, s')
ax2.set_ylabel('Frequency, Hz')

ax2.margins(x=0)
ax3 = fig.add_axes([0.9, 0.11, 0.015, 0.35])
plt.colorbar(mappable=cax, cax=ax3)
ax3.set_ylabel('Relative Amplitude, dB')

plt.savefig('blast3_spec.png', dpi=400)