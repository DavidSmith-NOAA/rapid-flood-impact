# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:41:59 2025

@author: david.smith
"""

#%%

import requests
import xarray as xr
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import zipfile
import os

#import matplotlib.pyplot as plt
#import cartopy
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#import pandas as pd # Assuming merged_impacts_table is a pandas DataFrame
#import numpy as np # For np.nanmax if you need to calculate max_streamflow for scaling
import plotly.express as px
from dash import Dash, dcc, html

import geopandas as gpd
from shapely.geometry import Point
import shutil # For zipping the shapefile components
#%%
# Set working directory unique to the user
# This path needs to be valid on the system where the script is run.
wdir = "C:/Users/david.smith/Documents/Python/rapid-flood-impact"

# Change current working directory to wdir for easier file management
try:
    os.chdir(wdir)
    print(f"Changed working directory to: {wdir}")
except FileNotFoundError:
    print(f"Warning: Directory '{wdir}' not found. Please ensure the path is correct.")
    print("Proceeding without changing directory. Ensure 'flood_points_with_max_streamflow_5hr_window.csv' is in the current script's directory.")
except Exception as e:
    print(f"An error occurred while changing directory: {e}")

# Load the impacts table
try:
    impacts = pd.read_csv('flood_points_with_max_streamflow_5hr_window.csv')
    print(f"Successfully loaded 'flood_points_with_max_streamflow_5hr_window.csv'. Shape: {impacts.shape}")
    if 'ID' not in impacts.columns:
        print("Warning: 'ID' column not found in 'impacts' table. Merge might fail or produce unexpected results.")
except FileNotFoundError:
    print("Error: 'flood_points_with_max_streamflow_5hr_window.csv' not found. Please ensure the file exists in the specified working directory or the script's directory.")
    impacts = pd.DataFrame() # Create an empty DataFrame to prevent errors later
except Exception as e:
    print(f"An error occurred while loading 'flood_points_with_max_streamflow_5hr_window.csv': {e}")
    impacts = pd.DataFrame() # Create an empty DataFrame


def get_max_streamflow_table_from_nwm_files(base_url, file_list):
    """
    Downloads NetCDF files from a given URL, extracts 'streamflow' and 'velocity' data,
    and creates a table with the maximum values for each link_id (feature_id)
    across all processed forecast hours.

    Args:
        base_url (str): The base URL where the files are located.
        file_list (list): A list of filenames to download and process.

    Returns:
        pd.DataFrame: A DataFrame containing 'feature_id', 'max_streamflow',
                      and 'max_velocity' (if available), or an empty DataFrame
                      if no data was processed.
    """
    all_forecast_data = [] # List to store data from each forecast hour

    print(f"Starting to process {len(file_list)} files from {base_url}")

    for filename in file_list:
        file_url = f"{base_url}/{filename}"
        local_filepath = os.path.join(os.getcwd(), filename) # Save to current working directory

        # Extract forecast hour from filename (e.g., f001 -> 1)
        try:
            # Assumes filename format like 'nwm.t14z.short_range.channel_rt.f001.conus.nc'
            forecast_hour_str = filename.split('.f')[1].split('.')[0]
            forecast_hour = int(forecast_hour_str)
        except (IndexError, ValueError):
            print(f"Could not parse forecast hour from filename: {filename}. Skipping.")
            continue

        print(f"\n--- Processing {filename} (Forecast Hour: {forecast_hour}) ---")
        print(f"Attempting to download from: {file_url}")

        try:
            # Download the file
            response = requests.get(file_url, stream=True)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            with open(local_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded {filename} to {local_filepath}")

            # Open the NetCDF file with xarray
            with xr.open_dataset(local_filepath) as ds:
                current_file_data = {}
                # Check for 'feature_id' variable/dimension which serves as the link ID
                if 'feature_id' in ds.variables:
                    current_file_data['feature_id'] = ds['feature_id'].values
                elif 'feature_id' in ds.dims: # If feature_id is a dimension but not a variable
                    current_file_data['feature_id'] = ds.coords['feature_id'].values
                else:
                    print(f"Warning: 'feature_id' not found as variable or dimension in {filename}. Skipping this file.")
                    continue

                # Extract streamflow data
                if 'streamflow' in ds.variables:
                    current_file_data['streamflow'] = ds['streamflow'].values
                else:
                    print(f"Warning: 'streamflow' variable not found in {filename}. Skipping this file.")
                    continue # Streamflow is mandatory for the table

                # Extract velocity data if available, otherwise fill with NaN
                if 'velocity' in ds.variables:
                    current_file_data['velocity'] = ds['velocity'].values
                else:
                    print(f"Note: 'velocity' variable not found in {filename}. Velocity column will have NaN for this file's data.")
                    # Fill with NaN values, ensuring it has the same shape as streamflow
                    current_file_data['velocity'] = np.full_like(current_file_data['streamflow'], np.nan)

                # Create a temporary DataFrame for this specific forecast hour
                df_current_hour = pd.DataFrame({
                    'feature_id': current_file_data['feature_id'],
                    'streamflow': current_file_data['streamflow'],
                    'velocity': current_file_data['velocity']
                })
                df_current_hour['forecast_hour'] = forecast_hour # Add forecast hour for context, though not used in final aggregation
                all_forecast_data.append(df_current_hour)

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
        except FileNotFoundError:
            print(f"Error: File not found after download: {local_filepath}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {filename}: {e}")
        finally:
            # Clean up: remove the downloaded file to save space
            if os.path.exists(local_filepath):
                os.remove(local_filepath)
                print(f"Cleaned up: Removed {local_filepath}")

    if not all_forecast_data:
        print("No data was successfully processed from any file.")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=['feature_id', 'max_streamflow', 'max_velocity'])

    # Concatenate all hourly data into a single DataFrame
    combined_df = pd.concat(all_forecast_data, ignore_index=True)

    # Group by feature_id and find the maximum streamflow and velocity across all forecast hours
    # np.nanmax is used to correctly handle NaN values (e.g., if velocity was missing for some files)
    max_flow_table = combined_df.groupby('feature_id').agg(
        max_streamflow=('streamflow', np.nanmax),
        max_velocity=('velocity', np.nanmax)
    ).reset_index()

    # Replace infinite values (which can result from np.nanmax on all NaNs) with NaN for cleaner output
    max_flow_table['max_streamflow'] = max_flow_table['max_streamflow'].replace([-np.inf, np.inf], np.nan)
    max_flow_table['max_velocity'] = max_flow_table['max_velocity'].replace([-np.inf, np.inf], np.nan)

    return max_flow_table

if __name__ == "__main__":
    # Define the latency in hours. This is the offset from current UTC time
    # to find the most recent available model run.
    latency_hours = 2

    # Get the current UTC time
    now_utc = datetime.now(timezone.utc)

    # Calculate the model run time based on current UTC time and latency.
    # This assumes the model run 'z' time is `latency_hours` behind the current UTC hour.
    model_run_time_utc = now_utc - timedelta(hours=latency_hours)

    # Format the date for the URL (YYYYMMDD)
    zulu_date = model_run_time_utc.strftime("%Y%m%d")
    # Format the hour for the URL and filenames (HHz)
    zulu_hour = model_run_time_utc.strftime("%H") + "z"

    # Construct the base URL dynamically using the determined date and hour
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwm/prod/nwm.{zulu_date}/short_range/"

    # Construct the list of file names dynamically for forecast hours f001 to f018
    file_names = [
        f"nwm.t{zulu_hour}.short_range.channel_rt.f{i:03d}.conus.nc" for i in range(1, 19)
    ]

    print(f"Determined Zulu Date: {zulu_date}")
    print(f"Determined Zulu Hour (Model Run Time): {zulu_hour}")
    print(f"Constructed Base URL: {base_url}")
    print(f"Example File Name: {file_names[0]}")

    # Call the updated function to get the max streamflow and velocity table
    max_flow_velocity_table = get_max_streamflow_table_from_nwm_files(base_url, file_names)

    if not max_flow_velocity_table.empty and not impacts.empty:
        print("\n--- Max Streamflow and Velocity Table (per feature_id) ---")
        # Print the DataFrame, formatting floats to two decimal places and without the index
        print(max_flow_velocity_table.to_string(index=False, float_format="%.2f"))

        # Ensure 'ID' column in impacts and 'feature_id' in max_flow_velocity_table are of compatible types
        # This is a common source of merge issues if data types don't match.
        try:
            impacts['ID'] = impacts['ID'].astype(str)
            max_flow_velocity_table['feature_id'] = max_flow_velocity_table['feature_id'].astype(str)
            print("\nConverted 'ID' and 'feature_id' columns to string type for robust merging.")
        except Exception as e:
            print(f"Warning: Could not convert ID/feature_id to string for merging: {e}")
            print("Attempting merge with original data types.")


        # Merge the max_flow_velocity_table with the impacts table
        merged_impacts_table = pd.merge(
            impacts,
            max_flow_velocity_table,
            left_on='ID',
            right_on='feature_id',
            how='left' # Use a left merge to keep all rows from the impacts table
        )

        print("\n--- Merged Impacts Table with NWM Streamflow and Velocity Data ---")
        # Print the merged DataFrame, formatting floats to two decimal places and without the index
        #print(merged_impacts_table.to_string(index=False, float_format="%.2f"))
    elif impacts.empty:
        print("\nImpacts table is empty, cannot perform merge.")
    else:
        print("\nCould not generate a max streamflow and velocity table to merge with impacts data.")
        print("\nOriginal Impacts Table (no merge performed due to empty NWM data):")
        #print(impacts.to_string(index=False, float_format="%.2f"))


#merged_impacts_table.to_csv('merged_impacts_table.csv')
#%%
#Convert cubic meters per second to cubic feet per second
merged_impacts_table['max_fcst_streamflow_cfs'] = merged_impacts_table['max_streamflow']*35.315
#Convert meters per second to feet per second
merged_impacts_table['max_fcst_velocity_fs'] = merged_impacts_table['max_velocity']*3.281

#%%
#Calulate the percent change. 
merged_impacts_table['percent_diff'] = ((merged_impacts_table['max_fcst_streamflow_cfs']- merged_impacts_table['max_streamflow_window_cfs'])/(merged_impacts_table['max_streamflow_window_cfs']))*100

#%%
"""
# Create a figure and an axes object with a Plate Carree projection
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Set the extent to CONUS (approximate boundaries)
# Adjust these bounds if you need a tighter or wider view
ax.set_extent([-125, -65, 25, 50], crs=ccrs.PlateCarree())

# Add geographic features
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='darkgray') # Country borders
ax.add_feature(cfeature.STATES, linestyle='-', edgecolor='gray', linewidth=0.5) # US State borders
ax.add_feature(cfeature.LAKES, alpha=0.5, facecolor='lightcyan')
ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.8)

# Plot the points, color-coded by max_streamflow
# Use transform=ccrs.PlateCarree() to tell cartopy that the coordinates are lat/lon
scatter = ax.scatter(
    merged_impacts_table['LON'],
    merged_impacts_table['LAT'],
    c=merged_impacts_table['percent_diff'],
    cmap='viridis', # Colormap for streamflow (e.g., 'viridis', 'plasma', 'hot')
    s=merged_impacts_table['percent_diff'] / merged_impacts_table['percent_diff'].max() * 200 + 50, # Scale size by streamflow, add base size
    alpha=0.8,
    edgecolors='black', # Add black borders to points for better visibility
    linewidths=0.5,
    transform=ccrs.PlateCarree(),
    label='Max Streamflow'
)

# Add a color bar
cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.05, shrink=0.7)
cbar.set_label('Max Streamflow (cfs)')

# Add title and labels
ax.set_title(f'CONUS Flood Impacts with Streamflow Percent Differnce ({zulu_date} {zulu_hour})', fontsize=14)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)

# Add gridlines and labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xformatter = cartopy.mpl.ticker.LongitudeFormatter()
gl.yformatter = cartopy.mpl.ticker.LatitudeFormatter()

# Save the map
map_filename = f"conus_flood_impacts_map_{zulu_date}_{zulu_hour}.png"
plt.savefig(map_filename, dpi=300, bbox_inches='tight')
print(f"Map saved as: {map_filename}")
plt.show() # Display the plot
"""

#%%
#Filter data by flows greater than the flows during the past LSR
rapid_flood_impacts = merged_impacts_table[merged_impacts_table['percent_diff'] > 110]


#%%
#Filter data by forecast flows greater than 50cfs. Anything less might be less significant.
rapid_flood_impacts = rapid_flood_impacts[rapid_flood_impacts['max_fcst_streamflow_cfs'] > 50]

#%%

#Drop unnecessary columms

# Define the list of columns you want to keep
columns_to_keep = [
    'VALID',
    'VALID2',
    'LAT',
    'LON',
    'WFO',
    'TYPETEXT',
    'CITY',
    'COUNTY',
    'STATE',
    'SOURCE',
    'REMARK',
    'dist_to_flowline',
    'VALID_DT',
    'max_streamflow_window_cfs',
    'feature_id',
    'max_fcst_streamflow_cfs',
    'max_fcst_velocity_fs',
    'percent_diff'
]

# Create the new DataFrame with only the specified columns
# This assumes 'rapid_flood_impacts' is already a pandas DataFrame
rapid_flood_impacts = rapid_flood_impacts[columns_to_keep].copy()

#%%
"""
def create_and_save_plotly_map(dataframe: pd.DataFrame, file_name: str = "flood_impacts_map.html"):
    """
    #Creates a Plotly scatter plot on a tile map and saves it as an HTML file.

    #Args:
        #dataframe (pd.DataFrame): The DataFrame containing the flood impact data.
        #file_name (str): The name of the HTML file to save the map.
"""

    # Ensure required columns are present
    required_columns = [
        'VALID2', 'LAT', 'LON', 'WFO', 'TYPETEXT', 'CITY', 'COUNTY', 'STATE',
        'SOURCE', 'REMARK', 'dist_to_flowline', 'VALID_DT',
        'max_streamflow_window_cfs', 'feature_id', 'max_fcst_streamflow_cfs',
        'max_fcst_velocity_fs', 'percent_diff'
    ]
    for col in required_columns:
        if col not in dataframe.columns:
            print(f"Error: Missing required column '{col}' in the DataFrame.")
            return

    # Define the columns to show in the hover popup
    hover_data_columns = [
        'VALID2', 'LAT', 'LON', 'WFO', 'TYPETEXT', 'CITY', 'COUNTY', 'STATE',
        'SOURCE', 'REMARK', 'dist_to_flowline', 'VALID_DT',
        'max_streamflow_window_cfs', 'feature_id', 'max_fcst_streamflow_cfs',
        'max_fcst_velocity_fs', 'percent_diff'
    ]

    # Create the scatter mapbox plot
    fig = px.scatter_mapbox(
        dataframe,
        lat="LAT",
        lon="LON",
        zoom=3,  # Adjust zoom level as needed
        height=600,
        mapbox_style="carto-positron", # Choose a map style (e.g., "open-street-map", "carto-positron", "stamen-terrain")
        hover_name="CITY", # Primary name to show on hover
        hover_data=hover_data_columns, # Additional data to show on hover
        title="Rapid Flood Impacts Scatter Plot"
    )

    # Update layout for better appearance, especially full screen
    fig.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},
        hovermode="closest"
    )

    # Save the figure as an HTML file
    fig.write_html(file_name)
    print(f"Map saved successfully as '{file_name}'")


# Call the function to create and save the map
create_and_save_plotly_map(rapid_flood_impacts, "rapid_flood_impacts_map.html")

"""






#%%
rapid_flood_impacts.to_csv('rapid_flood_impacts.csv')



#%%
# 1. Create a GeoDataFrame from the rapid_flood_impacts DataFrame
# First, create a 'geometry' column with Point objects from 'LON' and 'LAT'
geometry = [Point(xy) for xy in zip(rapid_flood_impacts['LON'], rapid_flood_impacts['LAT'])]
gdf_rapid_flood_impacts = gpd.GeoDataFrame(rapid_flood_impacts, geometry=geometry, crs="EPSG:4326") # WGS84

# Define the output base name for the shapefile and the zip file
output_base_name = "rapid_flood_impacts_shapefile"
output_shapefile_path = output_base_name + ".shp"
output_zip_path = output_base_name + ".zip"

# Define a temporary directory to store shapefile components before zipping
# This helps keep your main working directory clean
temp_shapefile_dir = "temp_shapefile_components"
os.makedirs(temp_shapefile_dir, exist_ok=True) # Create the temporary directory if it doesn't exist

# 2. Save the GeoDataFrame to a shapefile within the temporary directory
# Geopandas will create multiple files for the shapefile (.shp, .shx, .dbf, .prj, etc.)
# within the specified directory.
gdf_rapid_flood_impacts.to_file(os.path.join(temp_shapefile_dir, output_shapefile_path), driver="ESRI Shapefile")
print(f"Shapefile components created in temporary directory: {temp_shapefile_dir}/")

# 3. Zip the contents of the temporary directory
# The `root_dir` argument for make_archive should be the directory containing the files to be zipped.
# The `base_dir` argument specifies a subdirectory within root_dir from which to start archiving.
# We want to zip the *contents* of `temp_shapefile_dir`.
shutil.make_archive(output_base_name, 'zip', root_dir=temp_shapefile_dir)
print(f"Zipped shapefile created: {output_zip_path}")

# 4. Clean up the temporary directory containing the individual shapefile components
try:
    shutil.rmtree(temp_shapefile_dir)
    print(f"Cleaned up temporary directory: {temp_shapefile_dir}")
except OSError as e:
    print(f"Error: {temp_shapefile_dir} : {e.strerror}")



#%%

# Create a 'geometry' column with Point objects from 'LON' and 'LAT'
geometry = [Point(xy) for xy in zip(rapid_flood_impacts['LON'], rapid_flood_impacts['LAT'])]
gdf_rapid_flood_impacts = gpd.GeoDataFrame(rapid_flood_impacts, geometry=geometry, crs="EPSG:4326") # WGS84

#%%

catastrophic_phrases = [
    "fatal", "rescue","swift", "evac", 
    "destroy", "washed away", "bridge", "major", "significant"
]

considerable_phrases = [
    "roads", "roadway", "impassable", "closed",
    "home", "house", "business", "flooded"
]

def classify_color(remark):
    remark = str(remark).lower()
    if any(p in remark for p in catastrophic_phrases):
        return "magenta"
    elif any(p in remark for p in considerable_phrases):
        return "darkblue"
    else:
        return "lightblue"

gdf_rapid_flood_impacts["symbol_color"] = gdf_rapid_flood_impacts["REMARK"].apply(classify_color)


#%%

# Define your output shapefile name (without extension)
output_base = "rapid_flood_impacts_output"
output_dir = "output_shapefile"

# Create an output directory if it doesn’t exist
os.makedirs(output_dir, exist_ok=True)

# Define full shapefile path (without .shp extension)
shapefile_path = os.path.join(output_dir, output_base + ".shp")

# --- Save to shapefile ---
gdf_rapid_flood_impacts.to_file(shapefile_path, driver="ESRI Shapefile")

# --- Create a ZIP archive containing all shapefile components ---
zip_path = output_base + ".zip"

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        file_path = os.path.join(output_dir, output_base + ext)
        if os.path.exists(file_path):
            zipf.write(file_path, os.path.basename(file_path))

print(f"✅ Shapefile successfully created and zipped: {zip_path}")

#%%
# =========================================================
# 1️⃣  Prepare Data
# =========================================================
gdf_rapid_flood_impacts["lon"] = gdf_rapid_flood_impacts.geometry.x
gdf_rapid_flood_impacts["lat"] = gdf_rapid_flood_impacts.geometry.y

gdf_rapid_flood_impacts["max_fcst_streamflow_cfs"] = gdf_rapid_flood_impacts["max_fcst_streamflow_cfs"].fillna(0)

# Normalize alpha based on streamflow
streamflow = gdf_rapid_flood_impacts["max_fcst_streamflow_cfs"]
alpha_scaled = (streamflow - streamflow.min()) / (streamflow.max() - streamflow.min() + 1e-9)
gdf_rapid_flood_impacts["alpha"] = 0.2 + 0.8 * alpha_scaled  # opacity between 0.2–1.0

# =========================================================
# 2️⃣  Create Plotly Figure
# =========================================================
import plotly.express as px

color_map = {
    "magenta": "magenta",
    "darkblue": "darkblue",
    "lightblue": "lightblue"
}

fig = px.scatter_map(
    gdf_rapid_flood_impacts,
    lat="lat",
    lon="lon",
    color="symbol_color",
    hover_name="REMARK",
    hover_data={
        "max_fcst_streamflow_cfs": True,
        "symbol_color": True,
        "lat": False,
        "lon": False,
    },
    zoom=4,
    center={"lat": 39.5, "lon": -98.35},
    size_max=12,
)

fig.update_traces(
    marker=dict(
        size=12,
        opacity=gdf_rapid_flood_impacts["alpha"],
        color=gdf_rapid_flood_impacts["symbol_color"].map(color_map),
    )
)

# Map layout
fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        center={"lat": 39.5, "lon": -98.35},
        zoom=4,
        bearing=0,
        pitch=0,
    ),
    margin={"r":0,"t":0,"l":0,"b":0},
)

# Enable scroll wheel zoom
config = {
    "scrollZoom": True,  # <-- key for mouse wheel zoom
    "doubleClick": "reset",
}

# =========================================================
# 3️⃣  Create Dash App
# =========================================================


app = Dash(__name__)

app.layout = html.Div([
    html.H3("Rapid Flood Impact Dashboard", style={"textAlign": "center"}),
    dcc.Graph(
        figure=fig,
        style={"height": "90vh"},
        config=config
    )
])

if __name__ == "__main__":
    app.run(debug=True)

# =========================================================
# 4️⃣  Save HTML & GeoJSON
# =========================================================

# Save static HTML version
fig.write_html("rapid_flood_impacts.html", include_plotlyjs="cdn", full_html=True)

#%%
# Define the output GeoJSON file name
output_geojson_name = "rapid_flood_impacts.geojson"

gdf_rapid_flood_impacts.to_file(output_geojson_name, driver="GeoJSON")


#%%