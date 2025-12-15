import os
import glob
import warnings
from typing import Dict, List, Union, Callable, Tuple, Any
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, Polygon
from sklearn.neighbors import KDTree

from pvlib.solarposition import get_solarposition
from pvlib.irradiance import get_extra_radiation as E0_pvlib
import pandas as pd

    
def select_non_urban_neighbors(
    grid_crop: gpd.GeoDataFrame,
    ucdb_polygon: Polygon,
    k_neighbors_per_urban_point: int = 3,
    k_final: int = 6
) -> gpd.GeoDataFrame:
    """
    Identifies and selects the 'k_final' closest non-urban grid points to the 
    urban boundary defined by ucdb_polygon.

    The method finds the nearest neighbors among non-urban points for every 
    urban point and then selects the top unique 'k_final' neighbors overall.

    Parameters:
    ----------
    grid_crop : gpd.GeoDataFrame
        GeoDataFrame of grid points cropped to the area of interest.
    ucdb_polygon : shapely.geometry.Polygon
        The urban boundary polygon used to distinguish urban/non-urban points.
    k_neighbors_per_urban_point : int, optional
        The number of closest non-urban neighbors to find for EACH urban point. Defaults to 3.
    k_final : int, optional
        The final total number of unique non-urban points to be returned. Defaults to 6.

    Returns:
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the 'k_final' selected non-urban grid points,
        including their shortest distance to the urban area ('distance_deg').
        Returns an empty DataFrame if no urban points are found.
    """

    # Determine Urban Grid Point Membership 
    try:
        # Calculate which grid points fall within the urban polygon (ucdb_polygon)
        grid_crop["inside"] = grid_crop.geometry.within(ucdb_polygon)
    except NameError:
        # This error occurs if ucdb_polygon is undefined, though type hinting aims to prevent this.
        print("ERROR: 'ucdb_polygon' is not defined. Cannot proceed with boundary calculation.")
        raise

    # --- 1. Separate Non-Urban (Outside) Points for KDTree Indexing ---
    outside_points = grid_crop[grid_crop["inside"] == False].copy()

    # Extract coordinates of the non-urban points to build the KDTree
    coords = np.column_stack([
        outside_points.geometry.x.values,
        outside_points.geometry.y.values
    ])
    
    # Check if there are any outside points to build the tree
    if outside_points.empty:
        print("Warning: No non-urban (outside) grid points available in the cropped area.")
        return gpd.GeoDataFrame()
        
    tree = KDTree(coords)

    # Identify Urban Grid Points (Query Points) 
    # Grid points that are INSIDE the urban boundary.
    inside_points = grid_crop[grid_crop["inside"] == True]

    if inside_points.empty:
        print("Warning: No grid points found inside the urban boundary (ucdb_polygon). Cannot select neighbors based on boundary proximity.")
        return gpd.GeoDataFrame()

    # Extract coordinates of the urban points (the query points)
    query_coords = np.column_stack([
        inside_points.geometry.x.values,
        inside_points.geometry.y.values
    ])

    # Query the KDTree 
    
    # Handle the case where k_neighbors_per_urban_point is greater than the total number of outside_points
    k_safe = min(k_neighbors_per_urban_point, len(outside_points))
    
    # Query the KDTree (which indexes the OUTSIDE points)
    dist_all, idx_all = tree.query(query_coords, k=k_safe)

    # Process and Select the Top Unique Closest Neighbors 

    distances = dist_all.ravel()
    indices_in_outside_points = idx_all.ravel()

    # Create a temporary DataFrame to hold the results (outside index and distance)
    temp_results = pd.DataFrame({
        'outside_index': indices_in_outside_points,
        'distance_deg': distances
    })
    
    # Convert index to string to ensure consistent grouping behavior
    temp_results['outside_index'] = temp_results['outside_index'].astype(str)

    # Find the unique non-urban point index and keep the MINIMUM distance recorded
    closest_unique_results = temp_results.loc[
        temp_results.groupby('outside_index')['distance_deg'].idxmin()
    ]

    # Sort by distance and select the top 'k_final' unique points
    # Convert the index back to integer for .iloc indexing
    final_indices_int = (
        closest_unique_results.sort_values(by='distance_deg')
        .head(k_final)['outside_index'].astype(int).values
    )

    # Retrieve the final GeoDataFrame rows using the indices
    closest_points = outside_points.iloc[final_indices_int]
    
    # Assign the final shortest distance back to the selected points
    final_distances = (
        closest_unique_results.sort_values(by='distance_deg')
        .head(k_final)['distance_deg'].values
    )
    # Use .loc with the index for correct assignment
    closest_points.loc[closest_points.index, "distance_deg"] = final_distances

    print(f"Selected {len(closest_points)} unique non-urban points closest to the UCDB boundary.")
    
    return closest_points


def extract_cordex_series(
    closest_points: gpd.GeoDataFrame,
    mapping_mod2obs: Dict[str, str],
    variables_obs: Dict[str, str],
    institution: str,
    RCM: str,
    ensemble: str,
    model: str,
    version: str,
    start_date: str,
    end_date: str,
    base_path: str = "/lustre/gmeteo/DATA/ESGF/REPLICA/DATA/cordex/output/EUR-11/",
) -> pd.DataFrame:
    """
    Extract time series from CORDEX NetCDF files for given observation points.
    Includes:
        - Filtering only NetCDF files whose year is 2000–2025.
        - Searching version directories (vYYYYMMDD) inside each variable folder.
        - Efficient nearest-gridpoint extraction.
        - Merging point-specific time series into a single DataFrame.

    Returns:
        A DataFrame indexed by (YYYYMMDD, HH:MM:SS).
    """

    # ---------------------------------------------------------
    # 1. Prepare inputs
    # ---------------------------------------------------------
    institution = institution.strip()
    RCM = RCM.strip()
    ensemble = ensemble.strip()
    model = model.strip()
    version = version.strip()
    base_path = base_path.rstrip("/")

    selected_lons = closest_points.geometry.x.values
    selected_lats = closest_points.geometry.y.values

    print("\n--- CORDEX Series Extraction Initiated ---")
    print(f"Processing {len(selected_lats)} observation points...")
    for i in range(len(selected_lats)):
        print(f"  P{i}: Lat={selected_lats[i]:.4f}, Lon={selected_lons[i]:.4f}")

    df_final = pd.DataFrame()
    time_slice = slice(start_date, end_date)

    warnings.filterwarnings("ignore", category=UserWarning, module="xarray")

    # Scenarios to process
    scenarios = ["historical", "rcp85"]

    # ---------------------------------------------------------
    # 2. Loop through CORDEX variables
    # ---------------------------------------------------------
    for var_mod, obs_name in mapping_mod2obs.items():
        print(f"\nProcessing Variable: {var_mod}  (Obs: {obs_name})")

        col_obs = variables_obs[obs_name]
        all_files = []

        # ---------------------------------------------------------
        # 3. Locate NetCDF files inside version folders v*
        # ---------------------------------------------------------
        for scenario in scenarios:
            # Build path to the variable folder
            base_dir = os.path.join(
                base_path,
                institution,
                RCM,
                scenario,
                ensemble,
                model,
                version,
                "3hr",
                var_mod,
            )

            if not os.path.isdir(base_dir):
                print(f"  WARNING: Directory does not exist: {base_dir}")
                continue

            # Identify version directories (vYYYYMMDD)
            version_dirs = sorted(
                d for d in os.listdir(base_dir) if d.startswith("v")
            )

            if not version_dirs:
                print(f"  WARNING: No version directories found in: {base_dir}")
                continue

            # Loop through version directories
            for vdir in version_dirs:
                version_path = os.path.join(base_dir, vdir)

                nc_files = glob.glob(os.path.join(version_path, "*.nc"))
                if not nc_files:
                    print(f"  WARNING: No NetCDF files in {version_path}")
                    continue

                # Filter by years inside filename
                for f in nc_files:
                    fname = os.path.basename(f)

                    # Expected: *_YYYYMMDD-YYYYMMDD.nc
                    try:
                        date_range = fname.split("_")[-1].replace(".nc", "")
                        start_str, _ = date_range.split("-")
                        year = int(start_str[:4])
                    except Exception:
                        print(f"  WARNING: Could not parse dates in filename: {fname}")
                        continue

                    if 2000 <= year <= 2025:
                        all_files.append(f)

        if not all_files:
            print("  ATTENTION: No files from 2000–2025. Skipping variable.")
            continue

        # ---------------------------------------------------------
        # 4. Open combined dataset
        # ---------------------------------------------------------
        try:
            ds = xr.open_mfdataset(
                all_files,
                combine="by_coords",
                engine="h5netcdf",
                parallel=False,
                chunks={"time": 48},  # ~3 days at 3-hour resolution
            )
        except Exception as e:
            print(f"  ERROR opening dataset for {var_mod}: {e}")
            continue

        # Time selection
        try:
            ds = ds.sel(time=time_slice)
        except Exception as e:
            print(f"  ERROR selecting time period for {var_mod}: {e}")
            ds.close()
            continue

        # ---------------------------------------------------------
        # Detect rotated or regular grid coordinates
        # ---------------------------------------------------------
        if "rlon" in ds.coords and "rlat" in ds.coords:
            x_sel, y_sel = "rlon", "rlat"
        elif "x" in ds.coords and "y" in ds.coords:
            x_sel, y_sel = "x", "y"
        else:
            print(
                f"  ERROR: Dataset for '{var_mod}' lacks required grid coordinates "
                "('rlon/rlat' or 'x/y'). Skipping."
            )
            ds.close()
            continue
        
        # ---------------------------------------------------------
        # Load latitude and longitude (multiple possible names)
        # ---------------------------------------------------------
        lat_candidates = ["lat", "latitude"]
        lon_candidates = ["lon", "longitude"]
        
        lat_grid = None
        lon_grid = None
        
        # Try all combinations and pick the first one that exists
        for lat_name in lat_candidates:
            if lat_name in ds.variables:
                lat_grid = ds[lat_name].load()
                break
        
        for lon_name in lon_candidates:
            if lon_name in ds.variables:
                lon_grid = ds[lon_name].load()
                break
        
        # If still missing → error and skip
        if lat_grid is None or lon_grid is None:
            print(
                f"  ERROR: Could not find latitude/longitude variables for '{var_mod}'. "
                f"Tried: {lat_candidates} and {lon_candidates}"
            )
            ds.close()
            continue

        coordinate_columns = ["lat", "lon", "latitude", "longitude", "x", "y", "rlat", "rlon"]
        # ---------------------------------------------------------
        # 5. Extract time series for each observation point
        # ---------------------------------------------------------
        for i in range(len(selected_lats)):
            lat0 = selected_lats[i]
            lon0 = selected_lons[i]
        
            print(f"  -> Extracting P{i} ({lat0:.4f}, {lon0:.4f})")
        
            # Squared distance function
            def _dist(latg, long, lat0, lon0):
                return (latg - lat0) ** 2 + (long - lon0) ** 2
        
            # Compute distance map
            dist = xr.apply_ufunc(
                _dist,
                lat_grid,
                lon_grid,
                lat0,
                lon0,
                dask="parallelized"
            )
        
            # Find nearest gridpoint
            try:
                min_idx = dist.argmin(dim=[y_sel, x_sel])
            except Exception as e:
                print(f"    ERROR computing nearest gridpoint for P{i}: {e}")
                continue
        
            # Extract the DataArray for that point
            try:
                da_p = ds[var_mod].isel({
                    y_sel: int(min_idx[y_sel]),
                    x_sel: int(min_idx[x_sel])
                }).load()
            except Exception as e:
                print(f"    ERROR extracting P{i}: {e}")
                continue
        
            # Convert to DataFrame and **keep only the variable column**
            df = da_p.to_dataframe(name=f"{col_obs}_P{i}")[ [f"{col_obs}_P{i}"] ]
        
            # Add temporal index columns
            df["YYYYMMDD"] = df.index.strftime("%Y%m%d")
            df["HH:MM:SS"] = df.index.strftime("%H:%M:%S")
            df = df.set_index(["YYYYMMDD", "HH:MM:SS"])
        
            # Merge into final DataFrame
            df_final = df if df_final.empty else df_final.join(df, how="outer")

        
        # Close dataset
        ds.close()


    # ---------------------------------------------------------
    # 6. Final formatting
    # ---------------------------------------------------------
    df_final = df_final.reset_index()
    df_final["Time_Index_Aux"] = pd.to_datetime(
        df_final["YYYYMMDD"] + " " + df_final["HH:MM:SS"],
        errors="coerce"
    )

    print("\n--- CORDEX Series Extraction Completed ---")
    return df_final



def E0(day_of_year):
    """Calcula la razón de la irradiancia solar respecto al valor promedio."""
    return E0_pvlib(day_of_year)

def dew_point(T_C, RH_pct):
    """Calcula el punto de rocío a partir de T_Air y RH (ejemplo simplificado)."""
    # Placeholder para el cálculo del punto de rocío
    return T_C - (14.55 + 0.114 * T_C) * (1 - 0.01 * RH_pct) + ((2.5 + 0.007 * T_C) * (1 - 0.01 * RH_pct))**3 + (15.9 + 0.117 * T_C) * (1 - 0.01 * RH_pct)**14

def kd_erbs(Kt):
    """Modelo de fracción difusa de Erbs (ejemplo simplificado)."""
    # Placeholder para el modelo de Erbs
    return np.clip(0.952 - 1.02 * Kt, 0.165, 0.8)


def kelvin_to_fahrenheit(k):
        return (k - 273.15) * 9/5 + 32

def pr_to_mph(pr):
    return pr * 3600.0   # mm/s → mm/h (OBS format uses mm/h)

def compute_dewpoint_f(tas_K, hurs):
    T = tas_K - 273.15
    RH = hurs / 100.0
    a, b = 17.62, 243.12
    gamma = np.log(RH) + (a * T / (b + T))
    Td = (b * gamma) / (a - gamma)
    return Td * 9/5 + 32

def compute_net_rad(rsds, rsus, rlds, rlus):
    return (rsds - rsus) + (rlds - rlus)

def erbs_partition(rsds, sza):
    # sza in degrees
    cosz = np.cos(np.deg2rad(sza))
    cosz = np.where(cosz < 0.01, 0.01, cosz)
    kt = np.clip(rsds / (1367 * cosz), 0, 5)
    idn = np.zeros_like(kt)
    diffuse = np.zeros_like(kt)

    cond1 = kt <= 0.22
    idn[cond1] = rsds[cond1] * (1 - 0.99 * kt[cond1])
    diffuse[cond1] = rsds[cond1] - idn[cond1]

    cond2 = (kt > 0.22) & (kt <= 0.8)
    idn[cond2] = rsds[cond2] * (1.188 - 2.272 * kt[cond2] + 9.473 * kt[cond2]**2 - 21.856 * kt[cond2]**3 + 14.648 * kt[cond2]**4)
    diffuse[cond2] = rsds[cond2] - idn[cond2]

    cond3 = kt > 0.8
    idn[cond3] = rsds[cond3] * 0.165
    diffuse[cond3] = rsds[cond3] - idn[cond3]

    return idn, diffuse

def solar_zenith_index(df):
    if "sza" in df.columns:
        return df["sza"].values
    elif "Solar_Zenith_Angle_deg" in df.columns:
        return df["Solar_Zenith_Angle_deg"].values
    else:
        return np.full(len(df), 50)  # fallback constant

"""
process_cordex_to_obs_like.py

A single-file module that converts CORDEX model output (mixed naming conventions,
3-hourly or hourly timesteps) into an "obs-like" table used for diagnostics and
forcing (i-Tree Cool Air style Weather.csv / Radiation.csv).

Main features:
- Detects and repairs datetime index from YYYYMMDD + HH:MM:SS columns if present.
- Finds and maps model columns using many possible name variants.
- Converts units (K->F, Pa->kPa, fraction->%, kg/m2/s -> mm/h, etc).
- Computes derived variables: dew point, net radiation.
- Computes shortwave diffuse (DHI) and direct (DNI) using Erbs + cloud adjustment,
  with solar geometry computed via pvlib if available.
- Handles 3-hourly -> hourly redistribution for radiation and precipitation so
  daily/hourly cycle comparisons with observations are consistent.
- Produces a DataFrame with columns named as in mapping_mod2obs (plus derived names).
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

# Try to import pvlib; not strictly required but recommended for correct solar geometry.
try:
    import pvlib
    from pvlib.solarposition import get_solarposition
    PVLIB_AVAILABLE = True
except Exception:
    PVLIB_AVAILABLE = False


def kd_erbs(Kt):
    """
    Erbs (1982) diffuse fraction, implemented vectorized.
    Input: Kt (clearness index) = GHI / G0h (extraterrestrial horizontal irradiance).
    Returns kd = DHI / GHI (fraction of global that is diffuse).
    """
    Kt = np.asarray(Kt, dtype=float)
    kd = np.zeros_like(Kt, dtype=float)

    m1 = Kt <= 0.22
    kd[m1] = 1.0 - 0.09 * Kt[m1]

    m2 = (Kt > 0.22) & (Kt <= 0.8)
    K = Kt[m2]
    kd[m2] = (0.9511
              - 0.1604 * K
              + 4.388 * K**2
              - 16.638 * K**3
              + 12.336 * K**4)

    m3 = Kt > 0.8
    kd[m3] = 0.165

    kd = np.where(Kt < 0, 1.0, kd)  # night or invalid -> all diffuse
    return kd


def eccentricity_factor(day_of_year):
    """Eccentricity correction factor for Earth-Sun distance (approx.)."""
    return 1.0 + 0.033 * np.cos(2 * np.pi * day_of_year / 365.0)


def find_input_column(mod_var_prefix: str, point_id: str, dataframe: pd.DataFrame):
    """
    Heuristic column finder: looks for exact <prefix>_P<id> then searches
    by a list of keywords for columns ending in _P{id}.
    Returns the column name or None.
    """
    short_name = f"{mod_var_prefix}_P{point_id}"
    if short_name in dataframe.columns:
        return short_name

    # Keywords expanded to catch typical naming variants in CORDEX/other files
    keywords = {
        'rsds': ['Radiation_Shortwave_Downwelling', 'rsds', 'rsd', 'SWdown', 'sw_down'],
        'rsus': ['Radiation_Shortwave_Upwelling', 'rsus', 'rsu', 'SWup', 'sw_up'],
        'rlds': ['Radiation_Longwave_Downwelling', 'rlds', 'rld', 'LWdown', 'lw_down'],
        'rlus': ['Radiation_Longwave_Upwelling', 'rlus', 'rlu', 'LWup', 'lw_up'],
        'tas':  ['Temperature', 'tas', 'air_temperature', 't2m', 'Near-Surface'],
        'sfcWind': ['Wind', 'Speed', 'sfcWind', 'wind_speed', 'ws'],
        'ps': ['Pressure', 'ps', 'surface_air_pressure', 'p'],
        'hurs': ['Humidity', 'Relative', 'hurs', 'rh'],
        'pr': ['Precipitation', 'Rain', 'pr', 'precip'],
        'clt': ['Cloud', 'clt', 'cloud_fraction', 'cloud'],
    }
    search_terms = keywords.get(mod_var_prefix, [mod_var_prefix])

    # Candidate columns that end with _P{point_id}
    possible_cols = [c for c in dataframe.columns if c.endswith(f"_P{point_id}")]

    for term in search_terms:
        # case-insensitive containment
        matches = [c for c in possible_cols if term.lower() in c.lower()]
        # Special: for tas exclude dewpoint-like names
        if mod_var_prefix == 'tas':
            matches = [c for c in matches if 'dew' not in c.lower()]

        if matches:
            return matches[0]

    return None


def compute_dewpoint_from_T_RH(t_celsius: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """
    Magnus formula for dew point (returns Celsius).
    Input: t_celsius (°C), rh_pct (%)
    """
    a = 17.27
    b = 237.7
    # Protect values and convert rh to fraction
    rh_frac = (rh_pct / 100.0).clip(1e-6, 1.0)
    alpha = np.log(rh_frac) + (a * t_celsius) / (b + t_celsius)
    td_c = (b * alpha) / (a - alpha)
    return td_c


def compute_DHI_DNI_from_rsds(rsds: pd.Series,
                              rsus: pd.Series,
                              clt: pd.Series,
                              time_mid: pd.Series,
                              lat: Optional[float] = None,
                              lon: Optional[float] = None) -> pd.DataFrame:
    """
    Compute diffuse (DHI) and direct (DNI) from rsds with Erbs + cloud adjustment.
    - time_mid: pandas.DatetimeIndex or Series with mid-point times of 3-hour blocks.
    - clt: cloud fraction in percent (0..100) or unitless (0..1)
    - If pvlib is available and lat/lon provided, uses pvlib solar position to get zenith.
      Otherwise falls back to a crude cosine-of-zenith estimate using day-of-year and hour.
    Returns DataFrame columns: DHI, DNI, kd_erbs, kd_final, cos_theta, zenith, G0h, Kt
    """
    df = pd.DataFrame(index=time_mid.index)
    df['rsds'] = rsds
    # Ensure cloud fraction in 0..1
    clt_frac = clt.copy()
    if clt_frac.max() > 1.5:
        clt_frac = (clt_frac / 100.0).clip(0, 1)
    else:
        clt_frac = clt_frac.clip(0, 1)
    df['clt'] = clt_frac

    # Compute solar geometry
    df['doy'] = time_mid.dt.dayofyear
    df['hour'] = time_mid.dt.hour + time_mid.dt.minute / 60.0

    if PVLIB_AVAILABLE and (lat is not None) and (lon is not None):
        # Use pvlib to get accurate zenith
        solpos = get_solarposition(time_mid, latitude=lat, longitude=lon)
        df['zenith'] = solpos['zenith']
        df['cos_theta'] = np.cos(np.deg2rad(df['zenith']))
    else:
        # Fallback: approximate zenith based on hour and latitude. This is imperfect.
        # Use simple cosine daily cycle: zenith ~ arccos(cos(hour_angle)*sin(...)) is not trivial;
        # We'll compute a crude cos_theta to avoid divide-by-zero and allow kd calculation.
        # NOTE: user is strongly recommended to supply lat/lon and install pvlib for production.
        # Build a coarse diurnal cosine: maximum at noon, zero at night
        hour_angle = (df['hour'] - 12.0) * (np.pi / 12.0)  # radians
        df['cos_theta'] = np.maximum(0.0, np.cos(hour_angle))
        # Provide a placeholder zenith
        df['zenith'] = np.degrees(np.arccos(df['cos_theta']))

    # Extraterrestrial horizontal irradiance G0h = I_sc * E0 * cos_theta
    I_sc = 1367.0
    df['E0'] = eccentricity_factor(df['doy'])
    df['G0h'] = I_sc * df['E0'] * df['cos_theta']

    eps = 1e-6
    df['Kt'] = np.where(df['G0h'] > eps, df['rsds'] / df['G0h'], -1.0)
    df['kd_erbs'] = kd_erbs(df['Kt'])

    # METSTAT-like cloud adjustment: more cloud -> more diffuse; simple lerp between erbs and all-diffuse
    df['kd_final'] = (1.0 - df['clt']) * df['kd_erbs'] + df['clt'] * 1.0

    df['DHI'] = df['kd_final'] * df['rsds']
    # compute DNI safely: (GHI - DHI) / cos_theta (GHI is rsds here)
    df['DNI'] = np.where(df['cos_theta'] > 0, (df['rsds'] - df['DHI']) / df['cos_theta'], 0.0)
    df['DNI'] = df['DNI'].clip(lower=0.0)

    return df[['DHI', 'DNI', 'kd_erbs', 'kd_final', 'cos_theta', 'zenith', 'G0h', 'Kt']]


def process_cordex_to_obs_like(df_final: pd.DataFrame,
                               mapping_mod2obs: Dict[str, str],
                               lat: Optional[float] = None,
                               lon: Optional[float] = None) -> pd.DataFrame:
    """
    Main conversion function.

    Parameters:
    - df_final: DataFrame with model columns. Can include 'YYYYMMDD' + 'HH:MM:SS' columns,
      or already have a datetime index.
    - mapping_mod2obs: mapping from model short variable names to desired output long names.
      Example: {'rsds': 'Radiation_Shortwave_Downwelling_Wpm2(W/m^2)', ...}
    - lat, lon: optional floats used for solar geometry (recommended for DNI computation).

    Returns:
    - df_obs_like: DataFrame ready for diagnostics/plotting, with Time_Index_Aux and Time_Index.
    """
    df = df_final.copy()
    print(f"[INFO] Initial columns (sample): {df.columns[:8].tolist()}")

    # ---------------------------
    # 1) Build/repair datetime index
    # ---------------------------
    if 'YYYYMMDD' in df.columns and 'HH:MM:SS' in df.columns:
    
        # ---------------------------------------------------
        # FIX INVALID DATES (Option 3)
        # Example: replace 20000230 → 20000229 (leap year)
        # ---------------------------------------------------
        invalid_mask = (
            pd.to_datetime(
                df['YYYYMMDD'].astype(str) + ' ' + df['HH:MM:SS'].astype(str),
                format='%Y%m%d %H:%M:%S',
                errors='coerce'
            ).isna()
        )
    
        # Replace invalid dates with a valid one
        df.loc[invalid_mask, 'YYYYMMDD'] = 20000229
        # ---------------------------------------------------
    
        # Now build datetime safely
        time_series = df['YYYYMMDD'].astype(str) + ' ' + df['HH:MM:SS'].astype(str)
        new_index = pd.to_datetime(time_series, format='%Y%m%d %H:%M:%S', errors='coerce')
    
        df.index = new_index
        df = df.drop(columns=['YYYYMMDD', 'HH:MM:SS'])

    else:
        # If index isn't datetime, try to coerce
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors='coerce')
            except Exception as e:
                raise ValueError(f"Unable to parse index as datetime: {e}")

    # If there are still NaT in index, raise to let user inspect
    if df.index.isnull().any():
        raise ValueError("Datetime index contains NaT values after parsing. Please inspect input.")

    # Keep auxiliary time columns
    df['Time_Index_Aux'] = df.index
    df['Time_Index'] = df.index

    # ---------------------------
    # 2) Identify point IDs (P0, P1, ...)
    # ---------------------------
    p_ids = sorted(
        list({col.split("_P")[-1] for col in df.columns if "_P" in col and col.split("_P")[-1].isdigit()}),
        key=lambda x: int(x)
    )
    if not p_ids:
        # No point-suffixed columns; treat entire dataset as single point (P0)
        p_ids = ['0']

    # Output dataframe with time columns
    df_obs_like = pd.DataFrame(index=df.index)
    df_obs_like['Time_Index_Aux'] = df.index
    df_obs_like['Time_Index'] = df.index

    # ---------------------------
    # 3) Loop points and map variables
    # ---------------------------
    for p in p_ids:
        print(f"[INFO] Processing point P{p}")
        for mod_var, obs_name in mapping_mod2obs.items():
            # special: do not create 'rns' / 'tdps' output directly here (they are calculated)
            if mod_var in ['rns', 'tdps']:
                continue

            col_out = f"{obs_name}_P{p}"
            col_in = find_input_column(mod_var, p, df)

            if col_in is None:
                # warn and continue
                print(f"  [WARN] input column for '{mod_var}' not found for P{p}")
                continue

            series = df[col_in].astype(float)

            # --- Unit conversions ---
            # Temperature: model typically K; we want Fahrenheit output in obs-like file
            if mod_var == "tas":
                mean_val = series.mean(skipna=True)
                if mean_val > 200.0:  # Kelvin
                    series_f = (series - 273.15) * 9.0/5.0 + 32.0
                elif mean_val <= 60.0 and mean_val >= -50.0:
                    # uncertain: if ~20 -> assume Celsius
                    if mean_val <= 50.0:
                        series_f = series * 9.0/5.0 + 32.0
                    else:
                        # already Fahrenheit-ish
                        series_f = series
                else:
                    series_f = series
                df_obs_like[col_out] = series_f

            # Precipitation: model may provide kg m-2 s-1 (~mm/s). We'll convert to mm/h.
            elif mod_var in ["pr", "prr"]: 
                # If very small mean -> likely mm/s or kg/m2/s
                mean_val = series.mean(skipna=True)
                if mean_val < 0.1:
                    # treat as mm/s; convert to mm/h
                    series_mm_per_h = series * 3.6
                else:
                    # already mm/h or comparable; keep as-is
                    series_mm_per_h = series
                df_obs_like[col_out] = series_mm_per_h

            # Pressure: Pa -> kPa
            elif mod_var in ["ps", "psl"]:
                mean_val = series.mean(skipna=True)
                if mean_val > 80000.0:
                    df_obs_like[col_out] = series / 1000.0  # Pa -> kPa
                else:
                    df_obs_like[col_out] = series

            # Humidity: 0..1 -> 0..100
            elif mod_var == "hurs":
                if series.max(skipna=True) <= 1.0:
                    df_obs_like[col_out] = series * 100.0
                else:
                    df_obs_like[col_out] = series

            # Wind & radiation & cloud: copy directly (further processing below)
            else:
                df_obs_like[col_out] = series

        # ---------------------------------------------------------
        # Derived variables per point
        # ---------------------------------------------------------
        # Dew point: requires tas and hurs
        tas_in = find_input_column("tas", p, df)
        hurs_in = find_input_column("hurs", p, df)
        if tas_in and hurs_in:
            t_series = df[tas_in].astype(float)
            rh_series = df[hurs_in].astype(float)
            # Convert t_series to Celsius for dewpoint calculation
            mean_t = t_series.mean(skipna=True)
            if mean_t > 200:  # Kelvin
                t_c = t_series - 273.15
            elif mean_t > 50:  # likely Fahrenheit
                t_c = (t_series - 32.0) * 5.0/9.0
            else:
                t_c = t_series  # already Celsius

            # Ensure RH in percent
            if rh_series.max(skipna=True) <= 1.0:
                rh_pct = rh_series * 100.0
            else:
                rh_pct = rh_series

            td_c = compute_dewpoint_from_T_RH(t_c, rh_pct)
            # Convert to Fahrenheit output naming consistent with mapping
            df_obs_like[f"{mapping_mod2obs.get('tdps','Temperature_DewPoint_F(F)')}_P{p}"] = td_c * 9.0/5.0 + 32.0

        # Net radiation: if all four components exist, compute Net
        rsds_col = find_input_column("rsds", p, df)
        rsus_col = find_input_column("rsus", p, df)
        rlds_col = find_input_column("rlds", p, df)
        rlus_col = find_input_column("rlus", p, df)
        if all([rsds_col, rsus_col, rlds_col, rlus_col]):
            net_name = mapping_mod2obs.get('rns', 'Radiation_Net_Wpm2(W/m^2)')
            df_obs_like[f"{net_name}_P{p}"] = (df[rsds_col].astype(float)
                                               - df[rsus_col].astype(float)
                                               + df[rlds_col].astype(float)
                                               - df[rlus_col].astype(float))

        # Compute DHI/DNI (diffuse/direct) using Erbs + cloud correction if rsds exists
        if rsds_col:
            rsds_series = df[rsds_col].astype(float)
            # take cloud fraction if available, else zeros
            clt_col = find_input_column("clt", p, df)
            if clt_col:
                clt_series = df[clt_col].astype(float)
            else:
                clt_series = pd.Series(0.0, index=df.index)

            # Use the mid-time assumption for 3-hourly blocks (shift by 1.5 h)
            # If input freq appears to be 3-hourly, create time_mid offset for geometry.
            # Use df.index as the starting timestamp for each block.
            # time_mid = df.index + 1.5 hours (safe for 3h cadence)
            time_index = df.index
            # If time spacing is 3 hours, use midpoints
            freq_seconds = (time_index[1] - time_index[0]).total_seconds() if len(time_index) > 1 else 3600
            freq_hours = freq_seconds / 3600.0
            if np.isclose(freq_hours, 3.0):
                time_mid = time_index + pd.Timedelta(hours=1.5)
            else:
                # For hourly or other freq, use the timestamp as "mid"
                time_mid = time_index

            # Compute DNI/DHI table
            dhi_dni_df = compute_DHI_DNI_from_rsds(rsds_series, 
                                                   df[rsus_col].astype(float) if rsus_col else pd.Series(0.0, index=df.index),
                                                   clt_series,
                                                   pd.Series(time_mid, index=time_index),
                                                   lat=lat, lon=lon)
            # Write DHI and DNI with conventional names (short/diff)
            # Use the mapping names if user provided direct/diffuse mapping; otherwise create conventional names
            dhi_name = 'Radiation_Shortwave_Diffuse_downwelling_Wpm2(W/m^2)'
            dni_name = 'Radiation_Shortwave_Direct_downwelling_Wpm2(W/m^2)'
            df_obs_like[f"{dhi_name}_P{p}"] = dhi_dni_df['DHI'].values
            df_obs_like[f"{dni_name}_P{p}"] = dhi_dni_df['DNI'].values

    # ---------------------------
    # 4) Handle 3-hourly -> hourly redistribution & alignment
    # ---------------------------
    # Determine native model timestep (hours)
    if len(df.index) >= 2:
        dt_hours = (df.index[1] - df.index[0]).total_seconds() / 3600.0
    else:
        dt_hours = 1.0

    # If model is 3-hourly we will produce an hourly dataset by distributing 3-hourly
    # averages/fluxes into hourly values so that hourly and daily cycles are comparable.
    # Strategy:
    # - For radiative fluxes (rsds, rsus, rlds, rlus, computed DHI/DNI and net) and other
    #   flux-like variables, forward-fill across the interval (i.e., assume the same
    #   mean value applies to each contained hour) OR distribute evenly for precipitation.
    # - For precipitation: if original units were mm/h (after conversion above), and sampled
    #   every 3h, we assume the value represents an average rate over the 3-hour block, so
    #   we fill the same rate for the three contained hourly slots. If pr was originally a
    #   3-hour total (unknown), distribution equally across hours is used by dividing by 3.
    if np.isclose(dt_hours, 3.0):
        print("[INFO] Detected 3-hourly model timestep; resampling/interpolating to hourly for cycle computations.")

        # Identify columns that look like radiation, precipitation, etc. We'll attempt to resample every column,
        # but treat precipitation specially by dividing when needed.
        out_cols = df_obs_like.columns.tolist()
        hourly_index = pd.date_range(start=df_obs_like.index.min(), end=df_obs_like.index.max(), freq='H')

        # Create an hourly DataFrame and fill values appropriately
        hourly_df = pd.DataFrame(index=hourly_index)

        for col in out_cols:
            if col in ['Time_Index_Aux', 'Time_Index']:
                continue
            series = df_obs_like[col]
            # If the column is precipitation-like (by name)
            if 'Precipitation' in col or 'precip' in col.lower():
                # We assume series is in mm/h (after earlier conversion). If sampled every 3h,
                # set the hourly rate equal to the 3-hour rate (this preserves rate units).
                # If instead the 3-hour value is actually a total, this will overstate unless divided.
                # To be conservative, we'll distribute the value equally across the 3 contained hours:
                # i.e., hourly_rate = series / 1 (if value already mm/h) OR series/3 if it's a 3h total.
                # Heuristic: if mean < 0.5 mm/h treat as rate (so forward-fill). Otherwise, distribute.
                mean_val = series.mean(skipna=True)
                if mean_val > 0.5:
                    # one might interpret as 3h total; divide by 3 to get per-hour total
                    values_3h = series
                    # Build hourly series by reindexing to hourly and forward filling 3-hour blocks
                    temp = values_3h.reindex(hourly_index, method=None)  # NaNs at hours between 3h steps
                    # Fill each 3h slot with the 3h total divided by 3
                    # Find positions where original index exists
                    for t, v in values_3h.dropna().items():
                        # distribute v/3 into the three hours starting at t (t assumed to be start)
                        # if model timestamps correspond to the start of 3h block; if they are midpoints this will be off by 1.5h
                        hours = pd.date_range(start=t, periods=3, freq='H')
                        for h in hours:
                            if h in hourly_index:
                                temp.loc[h] = v / 3.0
                    hourly_df[col] = temp
                else:
                    # Very small mean -> treat as rate in mm/h and forward-fill for each hour of the block
                    temp = series.reindex(hourly_index, method='ffill')
                    hourly_df[col] = temp
            else:
                # Radiative / other variables: forward-fill the 3-hour value to each contained hourly slot
                temp = series.reindex(hourly_index, method='ffill')
                hourly_df[col] = temp

        # Re-add Time_Index columns
        hourly_df['Time_Index_Aux'] = hourly_df.index
        hourly_df['Time_Index'] = hourly_df.index

        # Replace df_obs_like with hourly_df for downstream cycle computations
        df_obs_like = hourly_df

    else:
        # For hourly or other frequencies, ensure Time_Index columns exist and are correct
        df_obs_like['Time_Index_Aux'] = df_obs_like.index
        df_obs_like['Time_Index'] = df_obs_like.index

    # Final cleanup: remove duplicate columns if any
    df_obs_like = df_obs_like.loc[:, ~df_obs_like.columns.duplicated()]

    # Return sorted columns for deterministic output
    cols_sorted = sorted([c for c in df_obs_like.columns if c not in ['Time_Index_Aux', 'Time_Index']])
    df_obs_like = df_obs_like[['Time_Index_Aux', 'Time_Index'] + cols_sorted]

    print("[INFO] process_cordex_to_obs_like: complete.")
    return df_obs_like

def compute_annual_and_daily_cycles(
    data_dir: str,
    variables_obs: Dict[str, str],
    filename_pattern: str = "df_*.csv"
) -> Tuple[
    Dict[str, Dict[str, Dict[str, pd.Series]]],
    Dict[str, Dict[str, Dict[str, pd.Series]]]
]:
    """
    Same behavior as before, but now cycles are computed PER MODEL.
    """

    # ------------------------------------------------------------
    # 1. Locate files
    # ------------------------------------------------------------
    search_path = os.path.join(data_dir, filename_pattern)
    files = glob.glob(search_path)

    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {search_path}")

    print(f"Found {len(files)} CSV files.")

    # Output dictionaries:
    # annual_cycles_mod_by_model[model][variable]["All_Points_Agg"]
    # daily_cycles_mod_by_model[model][variable]["All_Points_Agg"]
    annual_cycles_mod_by_model = {}
    daily_cycles_mod_by_model = {}

    # ------------------------------------------------------------
    # 2. PROCESS EACH MODEL SEPARATELY
    # ------------------------------------------------------------
    for f in files:

        # ------------------------------------------------------------------
        # Extract model name from CSV filename
        # Input format df_{institution}_{RCM}_{ensemble}_{model}_{version}.csv
        # ------------------------------------------------------------------
        base = os.path.basename(f)
        name_no_ext = base.replace(".csv", "")
        parts = name_no_ext.split("_")

        # df, institution, RCM, ensemble, model, version
        if len(parts) < 6:
            raise ValueError(f"Filename does not match expected pattern: {base}")

        model_name = "_".join(parts[1:6])  # institution_RCM_ensemble_model_version
        print(f"\nProcessing model: {model_name}")
        df_temp = pd.read_csv(f, index_col=0)

        # Fix time column naming
        if "Time_Index_Aux.1" in df_temp.columns:
            df_temp.rename(columns={"Time_Index_Aux.1": "Time_Index_Aux"}, inplace=True)
        elif "Time_Index_Aux.0" in df_temp.columns:
            df_temp.rename(columns={"Time_Index_Aux.0": "Time_Index_Aux"}, inplace=True)

        df_temp["Time_Index_Aux"] = pd.to_datetime(
            df_temp["Time_Index_Aux"], errors="coerce"
        )

        # Time of day
        df_temp["Time_of_Day"] = df_temp["Time_Index_Aux"].dt.strftime('%H:%M')

        # Allocate dictionaries for this model
        annual_cycles_mod_by_model[model_name] = {name: {} for name in variables_obs.keys()}
        daily_cycles_mod_by_model[model_name] = {name: {} for name in variables_obs.keys()}

        # ------------------------------------------------------------
        # Compute cycles per variable FOR THIS MODEL ONLY
        # ------------------------------------------------------------
        for name, obs_col in variables_obs.items():

            # P0–P8 columns
            model_cols = [f"{obs_col}_P{pid}" for pid in range(9)]
            model_cols_present = [c for c in model_cols if c in df_temp.columns]

            if not model_cols_present:
                print(f"Warning: {model_name}: No model columns for {name} ({obs_col})")
                continue

            # Aggregate P0–P8
            if name == "Precipitation Rate":
                aggregated_series = df_temp[model_cols_present].mean(axis=1)
            else:
                aggregated_series = df_temp[model_cols_present].mean(axis=1)

            key = "All_Points_Agg"

            # ----- Annual cycle -----
            month_index = df_temp["Time_Index_Aux"].dt.month

            if name == "Precipitation Rate":
                annual_cycles_mod_by_model[model_name][name][key] = \
                    aggregated_series.groupby(month_index).mean()
            else:
                annual_cycles_mod_by_model[model_name][name][key] = \
                    aggregated_series.groupby(month_index).mean()

            # ----- Daily cycle -----
            if name == "Precipitation Rate":
                daily_cycle_raw = aggregated_series.groupby(df_temp["Time_of_Day"]).mean()
            else:
                daily_cycle_raw = aggregated_series.groupby(df_temp["Time_of_Day"]).mean()

            # HH:MM -> decimal hour
            new_index = daily_cycle_raw.index.map(
                lambda t: int(t.split(":")[0]) + int(t.split(":")[1]) / 60.0
            )

            daily_cycles_mod_by_model[model_name][name][key] = pd.Series(
                daily_cycle_raw.values, index=new_index
            )

    print("\nFinished computing cycles PER MODEL.\n")
    return annual_cycles_mod_by_model, daily_cycles_mod_by_model

