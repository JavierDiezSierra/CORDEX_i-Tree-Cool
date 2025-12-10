import os
import pandas as pd
import geopandas as gpd

def read_weather(path):
    """Read weather CSV with flexible delimiter detection."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    # try comma then tab
    try:
        df = pd.read_csv(path, sep=",")
    except Exception:
        df = pd.read_csv(path, sep="\t")
    return df

def read_radiation(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=",")
    except Exception:
        df = pd.read_csv(path, sep="\t")
    return df


def load_ucdb_city(root, city):
    """
    Load and filter a city shapefile from the Urban Centre Database (UCDB).

    Parameters:
    root (str): The root directory where the shapefile is located.
    city (str): The name of the city to load.

    Returns:
    gpd.GeoDataFrame: A GeoDataFrame containing the filtered city shapefile.
    """
    ucdb_info = gpd.read_file(root + '/GHS_FUA_UCD/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg')
    ucdb_city = ucdb_info.query(f'UC_NM_MN =="{city}"').to_crs(crs='EPSG:4326')
    if city == 'London':
        ucdb_city = ucdb_city[ucdb_city['CTR_MN_NM'] == 'United Kingdom']
    if city == 'Santiago':
        ucdb_city = ucdb_city[ucdb_city['CTR_MN_NM'] == 'Chile']
    if city == 'Barcelona':
        ucdb_city = ucdb_city[ucdb_city['CTR_MN_NM'] == 'Spain']
    if city == 'Dhaka':
        ucdb_city = ucdb_city[ucdb_city['CTR_MN_NM'] == 'Bangladesh']
    if city == 'Naples':
        ucdb_city = ucdb_city[ucdb_city['CTR_MN_NM'] == 'Italy']
    return ucdb_city