import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from soundscape_psd_functions import load_stations_for_distance_calculation

stations = load_stations_for_distance_calculation(fullstations=True)

# Comparison stations
targets = {
    "PAIN": (63.73333, -148.91667),
    "PANN": (64.54796, -149.08398),
    "PATK": (62.32056, -150.09361),
}

# Haversine distance (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


sensors = {}
closest_sta = {}
for name, (lat, lon) in targets.items():
    distances = haversine(
        lat, lon,
        stations["Latitude"].values,
        stations["Longitude"].values
    )
    closest_indices = np.argsort(distances)[:6]
    closest_stations = stations.iloc[closest_indices]["Station"].values
    sensors[name] = closest_stations
    
    for idx in closest_indices:
        st = stations.iloc[idx]["Station"]
        closest_sta[st] = {"distance": float(distances[idx])}


N_BINS = 5

def parse_xml_file(station, channel="Z"):
    """
    Parse PSD XML for a station and return a DataFrame indexed by timestamps,
    with frequency and power columns suitable for median aggregation per wind bin.

    Returns:
        freqs: array of unique frequencies
        psd_df: DataFrame with index=timestamp, columns=freq, values=power
    """
    xml_file = f"data/psd_stations/psd_{station}_DP{channel}.xml"
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Dictionary to temporarily hold powers per timestamp
    psds_by_time = defaultdict(dict)

    all_freqs = set()

    # Iterate over each PSD block
    for psd in root.findall(".//Psd"):
        start_time_str = psd.get("start")
        start_dt = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))

        for value in psd.findall("value"):
            freq = float(value.get("freq"))
            power = float(value.get("power"))
            psds_by_time[start_dt][freq] = power
            all_freqs.add(freq)

    # Sort frequencies
    freqs = np.array(sorted(all_freqs))

    # Build DataFrame: rows = timestamps, columns = frequencies
    psd_df = pd.DataFrame(
        index=sorted(psds_by_time.keys()),
        columns=freqs,
        dtype=float
    )

    for time, freq_dict in psds_by_time.items():
        for f, p in freq_dict.items():
            psd_df.at[time, f] = p

    # Ensure index is datetime
    psd_df.index = pd.to_datetime(psd_df.index)

    return freqs, psd_df



for station, seismic in sensors.items():

    # -------------------------
    # Load WIND data
    # -------------------------
    wind_file = f"data/WIND/{station}.csv"

    wind = pd.read_csv(
        wind_file,
        comment="#",
        skiprows=[1]
    )

    wind["Date_Time"] = pd.to_datetime(wind["Date_Time"], errors="coerce")
    wind["wind_speed_set_1"] = pd.to_numeric(
        wind["wind_speed_set_1"], errors="coerce"
    )

    wind = wind.dropna(subset=["Date_Time", "wind_speed_set_1"])
    wind = wind.set_index("Date_Time")

    # Hourly mean wind
    wind_hourly = wind["wind_speed_set_1"].resample("1h").mean().dropna()

    # -------------------------
    # Create wind speed bins
    # -------------------------
    bins = pd.cut(
        wind_hourly,
        bins=[0, 1, 2, 3, 4, 5, np.inf],
        right=False
    )
    
    # Count PSDs in each bin
    bin_counts = bins.value_counts().sort_index()


    wind_binned = pd.DataFrame({
        "wind_speed": wind_hourly,
        "wind_bin": bins
    })
    for seis in seismic:
    # ---- Load PSD data ----
        freqs, psd_df = parse_xml_file(seis, channel="Z")

        # ---- Join PSD with wind bins ----
        # Align by timestamp
        psd_wind = psd_df.join(wind_binned, how="inner")

        # ---- Compute median PSD per wind bin ----
        median_psd = psd_wind.groupby("wind_bin", observed=False).median()
        # Keep only frequencies that exist in the PSD DataFrame columns
        common_freqs = sorted(set(freqs).intersection(psd_df.columns))
        median_psd = psd_wind.groupby("wind_bin", observed=False)[common_freqs].median()
        # Sort median PSD by wind bin
        median_psd = median_psd.sort_index()

        # Compute reference PSD (lowest wind bin)
        ref_psd = median_psd.iloc[0].values

        colors = [ '#999999','#377eb8', '#4daf4a', '#dede00', '#ff7f00', '#e41a1c','#f781bf']
        fig, ax = plt.subplots(2,1, sharex=True, figsize=(7,9))
        for i, (wind_bin, row) in enumerate(median_psd.iterrows()):
            shifted_psd = row.values - ref_psd
            ax[1].plot(
                common_freqs,
                shifted_psd,
                color=colors[i],
                label=(
                    f"{wind_bin.left:.0f} – {wind_bin.right:.0f} m/s [{bin_counts[wind_bin]}]"
                    if wind_bin.right != np.inf
                    else f" > {wind_bin.left:.0f} m/s [{bin_counts[wind_bin]}]"
                )
            )
            ax[0].plot(
                common_freqs,
                row.values,
                color=colors[i],
            )
            ax[0].set_ylabel("Power, dB", fontsize='x-large')
            ax[0].set_ylim(-150, -80)
            ax[1].set_ylabel("Power Difference, dB", fontsize='x-large')
            ax[1].set_xlabel("Frequency, Hz", fontsize='x-large')
            # set the tick values to be large
            ax[0].tick_params(axis='both', which='major', labelsize='large')
            ax[1].tick_params(axis='both', which='major', labelsize='large')
            ax[0].set_xscale("log")
            ax[1].set_xscale("log")
            ax[0].grid(True, which="both", alpha=0.3)
            ax[1].grid(True, which="both", alpha=0.3)
            # reverse legend entries
            handles, labels = ax[1].get_legend_handles_labels()
            ax[1].legend(handles[::-1], labels[::-1], fontsize='large')
            ax[0].set_xlim(5.2556e-3, 2.4355e2)
            ax[1].set_xlim(5.2556e-3, 2.4355e2)
            ax[0].set_title(f" {seis} distance to {station} = {closest_sta[seis]['distance']:.2f} km", fontsize='x-large')

        plt.tight_layout()
        plt.savefig(f"wind_plots/psd_wind_{station}_{seis}.png", dpi=400)
        plt.close()
