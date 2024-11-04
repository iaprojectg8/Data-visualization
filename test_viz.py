
import numpy as np
import os
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from shapely import Point
import mplcursors


def find_nearest(tree, gdf, lat, lon, ax, fig):
    distances, indices = tree.query([lat, lon], k=1)  # Find nearest point
    if type(distances) == float:
        distance, index = distances, indices
        closest_point = gdf.iloc[index]  # Get data for the closest point
        print(closest_point)
        ax.scatter(closest_point["LON"], closest_point["LAT"], marker="x")
    else:
        for distance, index in zip(distances, indices):
            print(distance, index)
            closest_point = gdf.iloc[index]  # Get data for the closest point
            print(closest_point)
            ax.scatter(closest_point["LON"], closest_point["LAT"], marker="x")


def check_file_coordinates(folder_name,shapes:gpd.GeoDataFrame):

    # Reproject the whole shape into a lat lon based one
    shapes = shapes.to_crs("EPSG:4326")
    result_dict = dict()
    for shape in shapes.iterrows():
        # Initialize distance and path
        distance_min = np.inf
        filepath = ""

        # Define the centroid of the shape
        geometry = shape[1].geometry
        centroid = geometry.centroid
        print(centroid)
        for file in tqdm(os.listdir(folder_name)):
            complete_path = os.path.join(folder_name, file)

            # Get a point object from the coordianates of the CSV
            df = pd.read_csv(complete_path)
            lat, lon = df.iloc[0]["lat"],  df.iloc[0]["lon"]
            df_point = Point(lon, lat)

            # Calculate a distance
            distance = centroid.distance(df_point)

            # Take the complete path for the closer point
            if distance<distance_min:
                distance_min = distance
                filepath = complete_path
        result_dict[distance_min] = filepath
    return result_dict[min(result_dict)]




def read_shape_file(shapefile_path):
    
    shapes = gpd.read_file(shapefile_path)
    print(shapes.shape)
    print(shapes)
    return shapes


def get_closer_point_from_shape_centroid():

    shape_path = "AOI 1/AOI.shp"
    folder_path = "Gambie calculation and request/Extended_Gambie_dataset"
    shapes = read_shape_file(shapefile_path=shape_path)
    csv_path = check_file_coordinates(folder_name=folder_path, shapes=shapes)
    return csv_path


def make_variation_heatmap_temperature(csv_path):
    
    # This path has been found with the function get_closer_point_from_shape_centroid
    df = pd.read_csv(csv_path)
    
    # Take only the temperature
    df = df[["date", "temperature_2m_mean"]]


    # Convert "date" to datetime and filter out 2050
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year <2020]



    # Create right decade next to data to which it corresponds
    df["decade_start"] = (df["date"].dt.year // 10) * 10
    df["decade_end"] = df["decade_start"] + 9
    df["decade"] = df["decade_start"].astype(str) + "-" + df["decade_end"].astype(str)



    # Extract month
    df["month"] = df["date"].dt.month

    # Overall and decade-specific monthly averages
    overall_monthly_mean = df.groupby("month")["temperature_2m_mean"].mean()
    decade_month_data = df.groupby(["decade", "month"]).agg({"temperature_2m_mean": "mean"}).reset_index()

    # Merge both of them and make the difference
    decade_month_data = decade_month_data.merge(overall_monthly_mean, on="month", suffixes=("_decade", "_overall"))
    decade_month_data["temp_diff"] = decade_month_data["temperature_2m_mean_decade"] - decade_month_data["temperature_2m_mean_overall"]



    # Pivot table for heatmap
    pivot_table = decade_month_data.pivot(index="decade", columns="month", values="temp_diff")[::-1]
    pivot_table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]



    # Plotting heatmap
    plt.figure(figsize=(9, 5))
    heatmap = sns.heatmap(pivot_table, annot=True, cmap="Reds", center=0, fmt=".2f", linewidths=0.5, cbar_kws={"label": "Temperature Difference"})
    heatmap.set_title("Monthly Temperature Differences by Decade")
    heatmap.set_xlabel("Month")
    heatmap.set_ylabel("Decade")
  
    plt.show()


def density_plot_tempereature(csv_path):

    df = pd.read_csv(csv_path)

    # Take only the temperature
    df = df[["date", "temperature_2m_mean"]]

    # Convert "date" to datetime and filter out 2050
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year <2040]

    df.set_index("date", inplace=True)
    df = df.resample("YE").mean()

    # Create period of 30 years
    df["30y_period_start"] = (df.index.year // 30) * 30
    df["30y_period_end"] = df["30y_period_start"] + 29
    df["30y_period"] = df["30y_period_start"].astype(str) + "-" + df["30y_period_end"].astype(str)

    unique_periods = df["30y_period"].unique()
    plt.figure(figsize=(10, 6))

    for period in unique_periods:
        period_data = df[df["30y_period"] == period]["temperature_2m_mean"]
        sns.kdeplot(period_data, label=period, fill=True, bw_adjust=2)  # Adjust `bw_adjust` for smoothness

    plt.xlabel("Temperature (°C)")
    plt.ylabel("Density")
    plt.title("KDE Plot of Temperature Distributions for Each 30-Year Period")
    plt.legend(title="30-Year Period")
    plt.show()


def make_variability_month_variability_decade(csv_path):
    

    # Maybe we will have to see later if it is possible to smooth the curve of the plot
    # This path has been found with the function get_closer_point_from_shape_centroid
    df = pd.read_csv(csv_path)
    
    # Take only the temperature
    df = df[["date", "temperature_2m_mean"]]


    # Convert "date" to datetime and filter out 2050
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year <2020]

    df["decade_start"] = (df["date"].dt.year // 10) * 10
    df["decade_end"] = df["decade_start"] + 9
    df["decade"] = df["decade_start"].astype(str) + "-" + df["decade_end"].astype(str)


    # Extract month
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # Overall and decade-specific monthly averages
    overall_monthly_mean = df.groupby("month")["temperature_2m_mean"].mean()
    overall_monthly_mean= pd.DataFrame(overall_monthly_mean)
    year_month_data = df.groupby(["decade", "year", "month"]).agg({"temperature_2m_mean": "mean"}).reset_index()
    
    # Merge both of them and make the difference
    year_month_data = year_month_data.merge(overall_monthly_mean, on="month", suffixes=("_year", "_overall"))
    year_month_data.to_csv("test.csv")

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=year_month_data, x='month', y='temperature_2m_mean_year', 
        hue='decade', palette='plasma_r', s=100, picker=True 
    )

    sns.lineplot(data=overall_monthly_mean, x="month", y="temperature_2m_mean", color="green", picker=False )
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(ticks=range(1, 13), labels=month_labels)

    # Add hover functionality
    cursor = mplcursors.cursor(hover=True)

    
    @cursor.connect("add")
    def on_add(sel):
        """
        Handles the event triggered when a data point is selected on the plot.
        Updates the annotation text and visual properties based on the selected index.

        Args:
            sel: The selection event object that provides the index of the selected data point 
                and access to the annotation object for modification.
        """
        if sel.index in year_month_data['year']:
            sel.annotation.set(
                text=f"Temperature: {round(year_month_data['temperature_2m_mean_year'][sel.index], 2)}\nYear: {year_month_data['year'][sel.index]} °C",
                multialignment="center",
                )
            sel.annotation.arrow_patch.set_visible(False) 
            sel.annotation.get_bbox_patch().set(fc="white", alpha=.9)
    
        else :
            # Here you can choose what information to show for the line plot.
            sel.annotation.set(text="Temperature trend line", horizontalalignment="center", verticalalignment="center")  # Example text for the line
            sel.annotation.arrow_patch.set_visible(False) 
            sel.annotation.get_bbox_patch().set(fc="white", alpha=.9)
    # Labels and title
    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")
    plt.title("Mean Temperature per Month, Colored by Decade")
    plt.legend(reverse=True)
    plt.show()





if "__main__":
    csv_path = csv_path = "cmip6_era5_data_daily_53.csv"
    # make_variation_heatmap_temperature(csv_path)
    # density_plot_tempereature(csv_path)
    make_variability_month_variability_decade(csv_path)