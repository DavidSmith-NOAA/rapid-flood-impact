# -*- coding: utf-8 -*-
"""
Created on Wed Jul 9 15:52:52 2025

Data Processing part of the Rapid Flood Impact Tool

@author: david.smith
"""


#%%

# Import Tools
import pandas as pd
import urllib.request
import requests
from datetime import datetime, timedelta
import tarfile
import os
import io # NEW: For in-memory file handling
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
import fiona
import matplotlib.pyplot as plt
from shapely.ops import nearest_points
import concurrent.futures
import threading
# import tempfile # No longer strictly needed if using BytesIO for NC files directly
from urllib3.exceptions import InsecureRequestWarning # To suppress warnings

# Suppress the InsecureRequestWarning globally for requests
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
#Start Time
s = datetime.now()
#%% Set working directory unique to the user
wdir = "C:/Users/david.smith/Documents/Python/rapid-flood-impact"

# Ensure the working directory exists
os.makedirs(wdir, exist_ok=True)
# Change current working directory to wdir for easier file management
os.chdir(wdir)
print(f"Working directory set to: {os.getcwd()}")

#%% Pull the recent LSRs for flooding
"""
# Number of days look back
look_back = 8962 # Approximately 24.5 years

# Select the End Date
date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

# Select the Start Date
date1 = (datetime.today() - timedelta(days=look_back)).strftime('%Y-%m-%d')

# Build URL and download
url = (
    'https://mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py?'
    f'sts={date1}T00:00Z&ets={date}T00:00Z&fmt=csv'
)
urllib.request.urlretrieve(url, 'report.csv')
"""
# Load CSV
df = pd.read_csv('lsr_report.csv', on_bad_lines='skip')
print(f"Initial df loaded. Shape: {df.shape}")

#Filter flood
df = df[df['TYPETEXT'].isin(['FLOOD', 'FLASH FLOOD'])]
#Filer to the CONUS only
df = df[~df['STATE'].isin(['HI','AK','PR', 'GU', 'AS'])]
print(f"df after filtering. Shape: {df.shape}")
print(f"df head:\n{df.head()}")

if df.empty:
    print("Error: DataFrame 'df' is empty after filtering. No flood events to process.")
    raise ValueError("No flood data found after filtering. Exiting.")

#%%
#Select random subset for testing (optional, uncomment for quick tests)
#df = df.sample(n=500, random_state=0)

#%% Extract and load hydrofabric data
fname = 'NWM_channel_hydrofabric.tar.gz'
hydrofabric_tar_path = os.path.join(wdir, fname)

gdb_dir_name = "NWM_v3_hydrofabric.gdb"
gdb_path = os.path.join(wdir, gdb_dir_name)

if not os.path.exists(gdb_path):
    if os.path.exists(hydrofabric_tar_path) and hydrofabric_tar_path.endswith(("tar.gz", ".tgz")):
        print(f"Extracting {fname}...")
        with tarfile.open(hydrofabric_tar_path, "r:gz") as tar:
            tar.extractall(wdir)
            print("Extraction complete.")
    else:
        print(f"Warning: Hydrofabric tarball '{fname}' not found at '{wdir}'. Please ensure it's there for extraction.")
        raise FileNotFoundError(f"Hydrofabric tarball '{fname}' not found. Cannot proceed without it.")

layers = fiona.listlayers(gdb_path)
print("Layers in GDB:", layers)

layer_name = 'nwm_reaches_conus'
flowlines = gpd.read_file(gdb_path, layer=layer_name)
flowlines = flowlines.to_crs(epsg=4326)

print("Flowlines columns:", flowlines.columns)
print(f"Flowlines loaded. Number of features: {len(flowlines)}")

#%% Convert flood report DataFrame to GeoDataFrame
flood_points = gpd.GeoDataFrame(
    df.copy(),
    geometry=[Point(xy) for xy in zip(df['LON'], df['LAT'])],
    crs='EPSG:4326'
)
print(f"flood_points GeoDataFrame created. Number of points: {len(flood_points)}")
print(f"flood_points columns: {flood_points.columns}")

#%% Project both to a suitable projected CRS for accurate distance
utm_crs = 'EPSG:26915'
flood_points_proj = flood_points.to_crs(utm_crs)
flowlines_proj = flowlines.to_crs(utm_crs)

#%% Nearest spatial join: points → nearest flowline
flowlines_proj['line_geom'] = flowlines_proj.geometry

flowlines_subset = flowlines_proj[['geometry', 'line_geom', 'ID']].copy()

joined = gpd.sjoin_nearest(
    flood_points_proj,
    flowlines_subset,
    how='left',
    distance_col='dist_to_flowline'
)

joined['snapped_geom'] = joined.apply(
    lambda r: nearest_points(r.geometry, r.line_geom)[1] if r.line_geom is not None else None,
    axis=1
)
joined = joined[joined['snapped_geom'].notna()]

#%%
# Set snapped points as active geometry and reproject
joined = joined.set_geometry('snapped_geom')
joined.set_crs(utm_crs, inplace=True)
flood_points = joined.to_crs("EPSG:4326")

flood_points = flood_points[flood_points['dist_to_flowline'] < 1100]
print(f"flood_points after distance filtering. Number of points: {len(flood_points)}")

if flood_points.empty:
    print("Error: All flood points were filtered out by the distance threshold. No data to process further.")
    raise ValueError("No flood data remaining after distance filtering. Exiting.")

#%%
# Prepare flood_points for date-based URL selection
flood_points['VALID_DT'] = pd.to_datetime(flood_points['VALID'], format='%Y%m%d%H%M')

OPERATIONAL_NWM_START_DATE = datetime(2023, 10, 1).date()
RETROSPECTIVE_NWM_3_0_START_DATE = datetime(1979, 2, 1).date()
RETROSPECTIVE_NWM_3_0_END_DATE = datetime(2023, 1, 31).date()

#%%
def get_nwm_url(date_obj):
    date_str = date_obj.strftime('%Y%m%d')
    hour_str = date_obj.strftime('%H')
    
    if date_obj.date() >= OPERATIONAL_NWM_START_DATE:
        base_url = f"https://nwcal-dstore.nwc.nws.noaa.gov/nwm/3.0/nwm.{date_str}/analysis_assim/"
        file_suffix = ".analysis_assim.channel_rt.tm00.conus.nc"
        url = f"{base_url}nwm.t{hour_str}z{file_suffix}"
    elif RETROSPECTIVE_NWM_3_0_START_DATE <= date_obj.date() <= RETROSPECTIVE_NWM_3_0_END_DATE:
        base_url = f"https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com/CONUS/netcdf/CHRTOUT/{date_str[:4]}/"
        retrospective_time_str = f"{date_str}{hour_str}00"
        file_suffix = ".CHRTOUT_DOMAIN1"
        url = f"{base_url}{retrospective_time_str}{file_suffix}"
    else:
        return None
    return url

#%%
def process_single_report_group(report_time_data, wdir_path):
    report_time, group = report_time_data
    results = []
    
    candidate_times = [
        report_time - timedelta(hours=2),
        report_time - timedelta(hours=1),
        report_time,
        report_time + timedelta(hours=1),
        report_time + timedelta(hours=2)
    ]

    streamflow_values_for_window = {}

    for current_time in candidate_times:
        url = get_nwm_url(current_time)

        if url is None:
            continue

        file_content_buffer = io.BytesIO() # In-memory buffer
        ds = None

        try:
            r = requests.get(url, stream=True, verify=False, timeout=120)
            if r.status_code == 200:
                for chunk in r.iter_content(chunk_size=8192):
                    file_content_buffer.write(chunk)
                file_content_buffer.seek(0) # Rewind for xarray
            elif r.status_code == 404:
                continue
            else:
                continue
        except requests.exceptions.Timeout:
            continue
        except requests.exceptions.RequestException as e:
            continue
        except Exception as e:
            print(f"An unexpected error occurred during download for {url}: {e}")
            continue

        try:
            # Open dataset directly from the in-memory buffer
            ds = xr.open_dataset(file_content_buffer)
            feature_ids = ds['feature_id'].values
            streamflow_data = ds['streamflow']

            for idx, row in group.iterrows():
                comid = int(row['ID'])

                if comid in feature_ids:
                    try:
                        if 'time' in streamflow_data.dims and streamflow_data['time'].size > 0:
                            flow = streamflow_data.sel(feature_id=comid).isel(time=0).values.item()
                        else:
                            flow = streamflow_data.sel(feature_id=comid).values.item()

                        if comid not in streamflow_values_for_window:
                            streamflow_values_for_window[comid] = []
                        streamflow_values_for_window[comid].append(flow)
                    except Exception as e:
                        print(f"⚠️ Error extracting streamflow for COMID {comid} at {current_time.strftime('%Y%m%d%H%M')}: {e}.")
        except Exception as e:
            print(f"⚠️ Error reading/processing NetCDF data from memory for {current_time.strftime('%Y%m%d%H%M')}: {e}. Skipping processing for this time.")
        finally:
            if ds is not None:
                ds.close()

    for idx, row in group.iterrows():
        comid = int(row['ID'])
        max_flow_val = None
        if comid in streamflow_values_for_window and streamflow_values_for_window[comid]:
            valid_flows = [f for f in streamflow_values_for_window[comid] if f is not None]
            if valid_flows:
                max_flow_val = max(valid_flows)
        results.append({'idx': idx, 'max_streamflow_window': max_flow_val})
    return results

def download_and_extract_streamflow_max_window_parallel(flood_points_df, num_workers=None):
    if num_workers is None:
        num_workers = min(32, os.cpu_count() * 2)
        print(f"Using {num_workers} workers for parallel processing.")

    grouped_data = list(flood_points_df.groupby('VALID_DT'))
    print(f"Processing {len(grouped_data)} unique report time groups in parallel.")

    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_group = {executor.submit(process_single_report_group, group_item, wdir): group_item
                           for group_item in grouped_data}

        for i, future in enumerate(concurrent.futures.as_completed(future_to_group)):
            try:
                all_results.extend(future.result())
                if (i + 1) % 10 == 0 or (i + 1) == len(grouped_data):
                    print(f"Completed {i + 1}/{len(grouped_data)} groups...")
            except Exception as exc:
                print(f'A group generated an exception: {exc}')

    flood_points_df['max_streamflow_window'] = None
    temp_series = pd.Series({res['idx']: res['max_streamflow_window'] for res in all_results})
    flood_points_df.loc[temp_series.index, 'max_streamflow_window'] = temp_series

    return flood_points_df

#%% Call the parallel function
flood_points = download_and_extract_streamflow_max_window_parallel(flood_points)

flood_points['max_streamflow_window_cfs'] = flood_points['max_streamflow_window'] * 35.315

print("\n--- Flood Points with Max Streamflow Data in Window ---")
print(flood_points[['VALID', 'TYPETEXT', 'geometry', 'ID', 'dist_to_flowline', 'max_streamflow_window', 'max_streamflow_window_cfs']].head())
print(f"\nTotal flood points processed: {len(flood_points)}")
print(f"Flood points with max streamflow data: {flood_points['max_streamflow_window'].count()}")

flood_points.to_csv('flood_points_with_max_streamflow_5hr_window.csv')


#%% 
#Calulate elapsed Time
e = datetime.now()

elapsed = (e - s).total_seconds()/60
print("Elapsed: ", elapsed, " min")