import sys
import fiona
import numpy as np
import pandas as pd
import geopandas as gpd
import xml.etree.ElementTree as ET

from pathlib import Path
from datetime import datetime
from shapely.geometry import Point
from collections import defaultdict

def ensure_project_root_on_sys_path(package_dir="soundscape", fallback_levels_up=2):
    """
    Ensure the project root (the directory that contains `package_dir`) is on sys.path.
    Returns the project root Path that was inserted.
    """
    _here = Path(__file__).resolve()
    project_root = None
    for p in _here.parents:
        if (p / package_dir).is_dir():
            project_root = p
            break
    if project_root is None:
        # fallback: assume `fallback_levels_up` levels up
        try:
            project_root = _here.parents[fallback_levels_up]
        except IndexError:
            project_root = _here.parents[-1]
    sys.path.insert(0, str(project_root))
    return project_root

#----------------------
#LOAD FILES
#----------------------

def load_stations_for_distance_calculation(fullstations=False):
    if fullstations:
        STATIONS_txt = "/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/full_nodes.txt"
    else:
        STATIONS_txt = "/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/parkshwy_nodes.txt"

    LAT_COL = "Latitude"
    LON_COL = "Longitude"
    stations_df = pd.read_csv(STATIONS_txt, sep="|")

    stations = gpd.GeoDataFrame(
        stations_df,
        geometry=[Point(xy) for xy in zip(stations_df[LON_COL], stations_df[LAT_COL])],
        crs="EPSG:4326"
    )
    TARGET_CRS = "EPSG:3338"   # Alaska Albers Equal Area
    stations = stations.to_crs(TARGET_CRS)
    return stations


# Function to parse a single XML file and return frequencies and powers
def parse_xml_file(station,channel="Z"):
    xml_file = f"data/psd_stations/psd_{station}_DP{channel}.xml"
    tree = ET.parse(xml_file)
    root = tree.getroot()
    values = root.findall(".//Psd//value")
    freqs = np.array([float(v.get("freq")) for v in values])
    powers = np.array([float(v.get("power")) for v in values])
    return station, freqs, powers


def parse_psds_grouped_by_day(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # day → list of (frequencies, powers)
    psds_by_day = defaultdict(lambda: {"freq": [], "power": []})

    for psd in root.findall(".//Psd"):
        start_time_str = psd.get("start")
        start_dt = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        day_key = start_dt.date()   # groups by yyyy-mm-dd

        # Extract all freq/power pairs for this PSD
        for value in psd.findall("value"):
            freq = float(value.get("freq"))
            power = float(value.get("power"))
            psds_by_day[day_key]["freq"].append(freq)
            psds_by_day[day_key]["power"].append(power)

    return psds_by_day


def load_kml_lat_lon(kml_file):

    LAT_COL = "Latitude"
    LON_COL = "Longitude"
    layers = fiona.listlayers(kml_file)
    print("KML layers found:", layers)

    gdfs = []
    for layer in layers:
        gdf = gpd.read_file(kml_file, layer=layer)
        if not gdf.empty:
            gdfs.append(gdf)

    type_dist = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")

    type_dist = type_dist[type_dist.geometry.notnull()]

    type_dist["geometry"] = type_dist.geometry.apply(
        lambda g: g if g.geom_type == "Point" else g.representative_point()
    )

    type_dist[LAT_COL] = type_dist.geometry.y
    type_dist[LON_COL] = type_dist.geometry.x

    print(f"Loaded {len(type_dist)} lat/lon points from {kml_file}")

    TARGET_CRS = "EPSG:3338"
    type_dist = type_dist.to_crs(TARGET_CRS)
    return type_dist


#----------------------
#CLACULATE MEDIANS
#----------------------

# Function to median the frequencies and powers
def median_data(frequencies, powers):
    freq_dict = {}
    for i, freq in enumerate(frequencies):
        if freq not in freq_dict:
            freq_dict[freq] = []
        freq_dict[freq].append(powers[i])
    med_powers = []
    for f in freq_dict.keys():
        med_powers.append(np.median(freq_dict[f]))
    return list(freq_dict.keys()), med_powers


# Compute daily median PSD
def compute_daily_medians(psds_by_day):
    daily_psd = {}

    for day, data in psds_by_day.items():
        freqs = data["freq"]
        powers = data["power"]

        med_freqs, med_powers = median_data(freqs, powers)
        daily_psd[day] = (med_freqs, med_powers)

    return daily_psd


def align_and_median_psd(station_freqs, station_powers, stations_list):
    """
    Align station PSDs to a common frequency grid and compute median PSD.

    Args:
        station_freqs (dict): mapping station -> 1D array of frequencies
        station_powers (dict): mapping station -> 1D array of median powers (same length as corresponding freqs)
        stations_list (iterable): ordered list/array of station identifiers

    Returns:
        all_freqs (ndarray): common sorted frequency grid
        aligned_powers (ndarray): shape (len(stations_list), len(all_freqs)) interpolated powers
        median_power (ndarray): median power across stations for each frequency
    """
    power_grid = np.zeros((len(stations_list), 125))  # will expand columns
    for i, station in enumerate(stations_list):
        station = str(station)
        _, p = median_data(station_freqs[station], station_powers[station])
        power_grid[i,:] = p
    median_power = np.median(power_grid, axis=0)
    return power_grid, median_power


def compute_station_low_noise_median(aligned_powers, n_low=10, ord=2):
    """
    For one station:
    - Rank spectra by norm (lowest noise first)
    - Take the n_low lowest-noise spectra
    - Return their median spectrum

    powers: array (n_spectra, n_freq)
    ord: norm order (2 = L2, 1 = L1, np.inf also valid)
    """
    # Compute norm for each spectrum
    noise_levels = np.linalg.norm(aligned_powers, axis=1, ord=ord)

    # Indices of lowest-noise spectra
    low_idx = np.argsort(noise_levels)[:n_low]

    # Median of the lowest-noise spectra
    low_noise_median = np.nanmedian(aligned_powers[low_idx, :], axis=0)
    print(len(low_noise_median))
    return low_noise_median

def station_low_noise_med_per_freq(station,channel="Z", n_low=10):
    """
    Return the median PSD (per frequency) of the
    10 lowest-norm 30-minute PSDs in an XML file.
    """
    xml_file = f"data/psd_stations/psd_{station}_DP{channel}.xml"
    tree = ET.parse(xml_file)
    root = tree.getroot()
    values = root.findall(".//Psd//value")

    f = np.array([float(v.get("freq")) for v in values])
    p_db = np.array([float(v.get("power")) for v in values])

    powers = {}
    for freq, power in zip(f, p_db):
        if freq not in powers:
            powers[freq] = []
        powers[freq].append(power)

    freqs = np.array(sorted(powers.keys()))

    low_noise_median = []
    for f in freqs:
        powers[f] = np.sort(powers[f])
        low_noise_median.append(np.median(powers[f][:n_low]))

    return station, freqs, low_noise_median
