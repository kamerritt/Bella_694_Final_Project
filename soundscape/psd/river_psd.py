import numpy as np
import pandas as pd
import geopandas as gpd
import xml.etree.ElementTree as ET

from matplotlib import pyplot as plt
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from soundscape_psd_functions import parse_xml_file, load_stations_for_distance_calculation, median_data, station_low_noise_med_per_freq


OUTPUT_csv = "data/stations_dist2river.csv"
if not Path(OUTPUT_csv).exists():
    stations = load_stations_for_distance_calculation(fullstations=True)
    GDB_PATH = "river_data/hydrusm010g.gdb"
    STREAM_LAYER = "Stream"

    # READ STREAMS (MOVING WATER) FROM FILE GDB
    streams = gpd.read_file(GDB_PATH,layer=STREAM_LAYER)
    print(f"Loaded {len(streams)} stream segments")

    # REPROJECT BOTH TO METERS (ALASKA)
    TARGET_CRS = "EPSG:3338"   # Alaska Albers Equal Area

    streams = streams.to_crs(TARGET_CRS)

    # DISTANCE TO NEAREST MOVING WATER (ANY STREAM)
    stream_union = streams.geometry.union_all()

    stations["dist_to_stream_m"] = stations.geometry.distance(stream_union)
    stations["dist_to_stream_km"] = stations["dist_to_stream_m"] / 1000.0

    # DISTANCE TO NENANA RIVER ONLY
    # Try to find a name field automatically
    name_fields = [c for c in streams.columns if "name" in c.lower()]

    if name_fields:
        name_field = name_fields[0]
        nenana = streams[streams[name_field].str.contains("Nenana", na=False)]

        if len(nenana) > 0:
            nenana_union = nenana.geometry.union_all()
            stations["dist_to_nenana_m"] = stations.geometry.distance(nenana_union)
            stations["dist_to_nenana_km"] = stations["dist_to_nenana_m"] / 1000.0
        else:
            print("No Nenana River found in name field")
    else:
        print("No river name field found — skipping Nenana-specific distance")

    # SAVE OUTPUTS
    # Keep only station ID and distance to stream in km
    output = stations[["Station", "dist_to_stream_km","dist_to_nenana_km"]]

    # Save to CSV
    output.to_csv(OUTPUT_csv, index=False)

# Load station data
seismo_data = pd.read_csv(OUTPUT_csv)

# Select stations >= 1500 and their distances to Nenana (exclude NaNs)
seismo_data['Station_num'] = pd.to_numeric(seismo_data['Station'], errors='coerce')
mask = (
    (seismo_data['Station_num'] >= 1500) &
    (seismo_data['Station_num'] <= 1590) &
    seismo_data['dist_to_nenana_km'].notna()
)
selected = seismo_data.loc[mask]

# stations as strings (matching later use) and distances as numeric array
stations = selected['Station_num'].astype(int).astype(str).values
distances = selected['dist_to_nenana_km'].values


# Step 1: Parallel XML parsing
station_freqs = {}
station_powers = {}

#with ProcessPoolExecutor() as executor:
#    for station, freqs, powers in executor.map(parse_xml_file, stations):
        #f, p = median_data(freqs, powers)
low = [False,True]
mad_yn = [False,True]
for low_noise in low:
    for supp in mad_yn:
        for i, station in enumerate(stations):
            station = str(station)
            if low_noise:
                sta, f, p = station_low_noise_med_per_freq(station)
            else:
                sta, frequencies, powers = parse_xml_file(station,channel="Z")
                f, p = median_data(frequencies, powers)
            station_freqs[station] = f
            station_powers[station] = p 

        aligned_powers = np.zeros((len(stations), 125))  # will expand columns
        aligned_powers_med = np.zeros((0, 125))  # will expand columns
        for i, station in enumerate(stations):
            station = str(station)
            f, p = median_data(station_freqs[station], station_powers[station])
            aligned_powers[i,:] = p
            if distances[i] > 0.46:
                aligned_powers_med = np.vstack((aligned_powers_med, p))
        med_power_all = np.median(aligned_powers_med, axis=0)

        #Plot the aligned powers minus the median power for distance ranges
        fig, ax = plt.subplots(figsize=(8, 6))
        power_diff_25 = np.zeros((0, 125))
        power_diff_32 = np.zeros((0, 125))
        power_diff_39 = np.zeros((0, 125))
        power_diff_46 = np.zeros((0, 125))
        power_diff_over = np.zeros((0, 125))

        for i, distance_km in enumerate(distances):
            power_stack = aligned_powers[i] - med_power_all
            #take median of powers for stations bellow 0.35 km, between 0.35km and 0.55, and above 0.55
            if distance_km < 0.25:
                power_diff_25 = np.vstack((power_diff_25, power_stack))
            elif distance_km < 0.32:
                power_diff_32 = np.vstack((power_diff_32, power_stack))
            elif distance_km < 0.39:
                power_diff_39 = np.vstack((power_diff_39, power_stack))
            elif distance_km < 0.46:
                power_diff_46 = np.vstack((power_diff_46, power_stack))
            elif distance_km >= 0.46:
                power_diff_over = np.vstack((power_diff_over, power_stack))

        groups = [power_diff_25, power_diff_32, power_diff_39, power_diff_46, power_diff_over]
        colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#999999']

        labels = ['< 0.25 km', '0.25 - 0.32 km', '0.32 - 0.39 km', '0.39 - 0.46 km', '>= 0.46km']

        # choose a reference frequency array from the loaded station_freqs (fallback to any available)
        if len(stations) > 0 and stations[0] in station_freqs:
            freq_ref = station_freqs[stations[0]]
        else:
            # fallback to the first entry in station_freqs
            freq_ref = next(iter(station_freqs.values()))

        for h, c, l in zip(groups, colors, labels):
            if h.size == 0:
                continue
            power_med = np.median(h, axis=0)
            power_mad = np.median(np.abs(h - power_med), axis=0)
            ax.plot(freq_ref, power_med, color=c, linewidth=1.5, label=f"{l} [{len(h)} sensors]")
            if supp:
                ax.fill_between(freq_ref, power_med - power_mad, power_med + power_mad, color=c, alpha=0.1, zorder=10)
                ax.plot(freq_ref, power_med - power_mad, color=c, linestyle='dashed', linewidth=0.7)
                ax.plot(freq_ref, power_med + power_mad, color=c, linestyle='dashed', linewidth=0.7)

        ax.set_xscale("log")
        ax.set_xlabel("Frequency, Hz", fontsize='x-large')
        ax.set_ylabel("Power Difference, dB", fontsize='x-large')
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(5.2556e-3, 2.4355e2)
        #Set y lim to only plot integer values
        ax.set_ylim(-1, 20)
        ax.set_yticks(np.arange(0, 21, 5))
        #make the tick labels larger on both axis
        ax.tick_params(axis='both', which='major', labelsize='large')
        # Place legend inside the plot at top-left
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize='large')
        fig.tight_layout()
        plt.savefig("river_psd{}{}.pdf".format("_low" if low_noise else "", "_mad" if supp else ""))