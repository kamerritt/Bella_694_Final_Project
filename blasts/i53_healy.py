from waveform_collection import gather_waveforms
from obspy.core import UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth
from array_processing.tools.plotting import array_plot
from lts_array import ltsva
import os
import matplotlib.pyplot as plt

start_times = [
    "2019-01-08 22:26",
    "2019-01-10 00:44",
    "2019-01-11 00:55",
    "2019-01-19 00:17",
    "2019-02-01 23:15",
    "2019-02-16 01:52",
    "2019-02-23 02:11",
    "2019-03-02 01:42",
    "2019-03-31 23:48",
    "2019-04-13 00:19"
]

blast_coords = [
    (-148.69, 63.92),
    (-148.86, 63.91),
    (-148.92, 63.91),
    (-148.71, 63.99),
    (-148.96, 63.92),
    (-148.66, 63.98),
    (-148.68, 63.97),
    (-148.76, 64.01),
    (-148.67, 63.95),
    (-148.74, 63.99)
]

for i, start_time in enumerate(start_times):
    # Data collection parameters
    SOURCE = 'IRIS'
    NETWORK = 'IM'
    STATION = 'I53H?'
    LOCATION = '*'
    CHANNEL = 'BDF'
    START = UTCDateTime(start_time)
    END = START + 10 * 60  # 10 minutes after START

    # Filtering
    FMIN = 1  # [Hz]
    FMAX = 3  # [Hz]

    # Array processing
    WINLEN = 30  # [s]
    WINOVER = 0.9

    # Grab and filter waveforms
    st = gather_waveforms(
        SOURCE, NETWORK, STATION, LOCATION, CHANNEL, START, END, remove_response=True
    )
    st.filter('bandpass', freqmin=FMIN, freqmax=FMAX, corners=2, zerophase=True)
    st.taper(max_percentage=0.01)

    latlist = [tr.stats.latitude for tr in st]
    lonlist = [tr.stats.longitude for tr in st]

    # Array process
    vel, baz, t, mdccm, stdict, sigma_tau, conf_int_vel, conf_int_baz = ltsva(
        st, latlist, lonlist, WINLEN, WINOVER, alpha=1
    )

    # Plot
    fig, axs = array_plot(st, t, mdccm, vel, baz, ccmplot=True)
    # blast_coords is a list of (lon, lat) tuples; use those values for target location
    true_baz = gps2dist_azimuth(latlist[0], lonlist[0], blast_coords[i][1], blast_coords[i][0])[1]
    axs[3].axhline(true_baz, zorder=-5, color='gray', linestyle='--')


    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"i53_healy_blast_{START.strftime('%Y%m%d_%H%M')}.pdf")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
