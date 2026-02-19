import obspy
import pandas as pd
import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt
from datetime import datetime, timezone
from scipy.signal import hilbert, savgol_filter


def steepest_slope_indices(x, ys):
    """
    Returns the index of steepest slope for each curve
    """
    # dy/dx for all curves
    slopes = np.gradient(ys, x, axis=1)

    # index of max |slope| per curve
    return np.argmax(np.abs(slopes), axis=1)


def align_curves_by_steepest_slope(x, ys):
    """
    Align curves so their steepest slopes line up
    """
    indices = steepest_slope_indices(x, ys)
    target_index = int(np.median(indices))

    aligned_ys = np.empty_like(ys)

    for i, idx in enumerate(indices):
        shift = target_index - idx
        aligned_ys[i] = np.roll(ys[i], shift)

    return aligned_ys, target_index


def median_curve(aligned_ys):
    """
    Median across aligned curves
    """
    return np.median(aligned_ys, axis=0)


def align_medians_using_existing_indices(median_A, median_B,
                                         align_idx_A, align_idx_B):
    """
    Align median_B to median_A using existing alignment indices
    """
    shift = align_idx_A - align_idx_B
    aligned_median_B = np.roll(median_B, shift)

    return median_A, aligned_median_B


def plot_aligned_curves(x, aligned_ys, median_y, align_idx,fig_idx, color_med):

    # plot all curves
    for y in aligned_ys:
        ax[fig_idx].plot(x, y, color='gray', alpha=0.3, linewidth=1)

    # plot median curve
    ax[fig_idx].plot(x, median_y, color=color_med, linewidth=2, label='Median curve')

    # mark alignment point
    ax[fig_idx].axvline(x[align_idx], color='k', linestyle='--',
                label='Steepest slope alignment')


signal_file = 'trains2019mar13_feb19_DPZ.csv'
trains_df = pd.read_csv(signal_file)

ns_timestamp = trains_df['timestamp_NS']
sn_timestamp = trains_df['timestamp_SN']
ns_timestamp_adjust = trains_df['NS_picks']
sn_timestamp_adjust = trains_df['SN_picks']
folder = '/scratch/irseppi/50sps/'

ns_envelopes = np.empty((0, 50001), int)
sn_envelopes = np.empty((0, 50001), int)

fig, ax = plt.subplots(figsize=(8, 5))
for timestamps, timestamp_adjusts, label in [
    (ns_timestamp, ns_timestamp_adjust, 'NS'),
    (sn_timestamp, sn_timestamp_adjust, 'SN')
    ]:

    for i, start in enumerate(timestamps):
        ht = datetime.fromtimestamp((start - timestamp_adjusts[i]), tz=timezone.utc)
        month = ht.month
        day = ht.day
        if len(str(day)) == 1:
            day = '0' + str(day)
        file = folder +'/2019_0'+ str(month) + '_' + str(day) +'/'+'ZE_1119_DPZ.msd'

        hour = ht.hour
        mins = ht.minute
        secs = ht.second
        # -75 - 60  for alignment on peaks amp
        if Path(file).exists():
            tr = obspy.read(file)
            if label == 'NS':
                tr[0].trim(
                    tr[0].stats.starttime + (hour * 3600) + (mins * 60) + secs - 100 - 75,
                    tr[0].stats.starttime + (hour * 3600) + (mins * 60) + secs + 900 - 75
                )
            elif label == 'SN':
                tr[0].trim(
                    tr[0].stats.starttime + (hour * 3600) + (mins * 60) + secs - 100,
                    tr[0].stats.starttime + (hour * 3600) + (mins * 60) + secs + 900
                )
            data = tr[0][:]
            t_wf = tr[0].times()

            amplitude_envelope = np.abs(hilbert(data))
            amplitude_envelope = savgol_filter(amplitude_envelope, len(amplitude_envelope)//20, 3)

            # store each envelope and its timebase
            if label == 'NS':
                ns_envelopes = np.vstack((ns_envelopes, amplitude_envelope))
            elif label == 'SN':
                sn_envelopes = np.vstack((sn_envelopes, amplitude_envelope))


aligned_ns, align_idx_ns = align_curves_by_steepest_slope(np.array(t_wf), ns_envelopes)
median_ns = median_curve(aligned_ns)

aligned_sn, align_idx_sn = align_curves_by_steepest_slope(np.array(t_wf), sn_envelopes)
median_sn = median_curve(aligned_sn)
ns_med, sn_med = align_medians_using_existing_indices(median_ns, median_sn,
                                        align_idx_ns, align_idx_sn)

ns_mad = np.median(np.abs(ns_envelopes - ns_med), axis=0)
sn_mad = np.median(np.abs(sn_envelopes - sn_med), axis=0)

time = np.array(t_wf) - 350

ax.plot(time, sn_med, label='Northbound [22 signals]', color='C0', linewidth=1.5)
ax.plot(time - 30 , ns_med, label='Southbound [22 signals]', color='C1', linewidth=1.5)
med_lines = True
if med_lines:
    ax.fill_between(time, sn_med - sn_mad, sn_med + sn_mad, color='C0', alpha=0.1, zorder=10)
    ax.fill_between(time - 30, ns_med - ns_mad, ns_med + ns_mad, color='C1', alpha=0.1, zorder=10)

    ax.plot(time, sn_med - sn_mad, color='C0', linestyle='dashed', linewidth=0.7)
    ax.plot(time, sn_med + sn_mad, color='C0', linestyle='dashed', linewidth=0.7)
    ax.plot(time - 30, ns_med - ns_mad, color='C1', linestyle='dashed', linewidth=0.7)
    ax.plot(time - 30, ns_med + ns_mad, color='C1', linestyle='dashed', linewidth=0.7)

ax.set_xlabel('Time, s', fontsize='x-large')
ax.set_ylabel('Median Amplitude Envelope, counts', fontsize='x-large')
ax.set_xlim(0, 500)
if med_lines:
    ax.set_ylim(0, 8000)
else:
    ax.set_ylim(0, 4300)
ax.legend(fontsize='large')
ax.grid()
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.savefig('ns_train_comp.pdf')
