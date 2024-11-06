
import numpy as np
import os
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from shapely import Point
import mplcursors
from scipy.stats import linregress
import matplotlib.cm as colormaps
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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

def get_period_trend(df : pd.DataFrame, start, stop):

    # Calculating the trend lines
    df = df[(df.index > start) & (df.index < stop)]
    years = df.index
    annual_temperatures = df["temperature_2m_mean"]
    slope, intercept, r_value, p_value, std_err = linregress(years,annual_temperatures)
    print(slope)
    # Create the trend line using the slope and intercept
    trend_line = slope * years + intercept

    return trend_line, years


def build_trend_plot(year_temperature_average_df, periods):
    cmap = plt.get_cmap('jet')
    cmap_percent = np.linspace(0.3, 0.9, num=len(periods))
    for i, period in enumerate(periods): 

        start, end = period.split("-")
        trend_line, years = get_period_trend(year_temperature_average_df, int(start), int(end))
        plt.plot(years, trend_line, color=cmap(cmap_percent[i]), label=f"{period} Trend")

def temperature_trends(csv_path):
    df = pd.read_csv(csv_path)
    
    # Take only the temperature
    df = df[["date", "temperature_2m_mean"]]


    # Convert "date" to datetime and filter out 2050
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year<2050]
    df["year"] = df["date"].dt.year
    year_temperature_average_df = pd.DataFrame(df.groupby("year")["temperature_2m_mean"].mean())
    year_temperature_average_df.to_csv("test.csv")

    # Create the figure
    plt.figure(figsize=(10, 6))
    # Here I can make two loops to take build and then add to the graph the different plots, with the legend automatically
    sns.lineplot(data=year_temperature_average_df, x="year", y="temperature_2m_mean", color="grey", label="Year Average Temperatures")
    # trend_line, years = get_period_trend(year_temperature_average_df, 1950, 2050)
    periods = ["1950-2050", "1970-2050", "1990-2050", "2010-2050"]
    build_trend_plot(year_temperature_average_df, periods)
    
    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")
    # plt.ylim(year_temperature_average_df["temperature_2m_mean"].min()-1, year_temperature_average_df["temperature_2m_mean"].max()+1)
    plt.title("Mean Year Temperature and trends from different periods")
    plt.legend()
    plt.show()

def make_monthly_standard_deviation_over_years(csv_path):
    df = pd.read_csv(csv_path)
    
    # Take only the temperature
    df = df[["date", "temperature_2m_mean"]]


    # Convert "date" to datetime and filter out 2050
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year<2020]

    # Step 1: Filter the data to include only the years from 1990 to 2020
    
    filtered_data = df[(df['date'].dt.year >= 1990) & (df['date'].dt.year <= 2020)]
    filtered_data["month"] = filtered_data["date"].dt.month

    # Calculate the montly means temperature through the references years
    monthly_means = filtered_data.groupby("month")["temperature_2m_mean"].mean()
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # Put the corresponding averages faced to the right month for each year
    df = df.merge(monthly_means.rename('monthly_mean'), on="month")
    # Step 1: Calculate the difference from the monthly means
    df['squared_diff'] = (df['temperature_2m_mean'] - df['monthly_mean'])**2
    month_std = pd.DataFrame(df.groupby(["year","month"])["squared_diff"].mean() ** (1 / 2))
    month_std = month_std.reset_index()
   

    pivot_table = month_std.pivot(index="year", columns="month", values="squared_diff")
    pivot_table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Create the figure
    plt.figure(figsize=(10, 6))
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


    conditions = [
        (month_std['squared_diff'] < 1),
        (month_std['squared_diff'] >= 1) & (month_std['squared_diff'] < 2),
        (month_std['squared_diff'] >= 2) & (month_std['squared_diff'] < 3),
        (month_std['squared_diff'] >= 3)
    ]
    size_categories = [50, 100, 200, 500]  # Corresponding sizes
    month_std['size_category'] = np.select(conditions, size_categories)

    color_labels = ["green", "gold", "darkorange","red"]
    month_std['color'] = np.select(conditions, color_labels)


    legends = ["SD < 1", "1 < SD < 2", "2 < SD < 3", "SD > 3"]
    month_std["legend"]=np.select(conditions, legends)
    month_std.to_csv("j.csv")

    # Plot with discrete sizes and colors

    print(month_std)
    scatter = sns.scatterplot(
        data=month_std, 
        x='month', 
        y='year', 
        hue='legend', 
        palette=["green", "gold", "darkorange","red"],  # Discrete colors
        hue_order=["SD < 1", "1 < SD < 2", "2 < SD < 3", "SD > 3"],
        size='size_category',  # Discrete sizes
        sizes=[50, 100, 200, 500], 
        alpha=0.5,
        # picker=True
    )

    cursor = mplcursors.cursor(hover=True)

    @cursor.connect("add")
    def on_add(sel):
        # Allows to see what is in the dictionary of the artist, which corresponds to the scatter
        print(sel.artist.__dict__)

        sel.artist.set_alpha(0.1)
        sel.artist.__dict__['_facecolors'][sel.index][3] = 1.

        # This was for the edge color to make it black
        # sel.artist.__dict__['_edgecolors'] = np.array([[0,0,0,1]])
    
        index = sel.index
        row = month_std.iloc[index]
        
        # Set the annotation for the selected point
        sel.annotation.set(
            text=f"SD: {round(row['squared_diff'], 2)}\nYear: {row['year']}",
            multialignment="center",
        )
        sel.annotation.arrow_patch.set_visible(False) 
        sel.annotation.get_bbox_patch().set(fc="white", alpha=.9)

    # @cursor.connect("remove")
    # def on_remove(sel):
    #     print("in the remove function")
    #     sel.artist.set_alpha(0.5)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_labels[i], markersize=10, alpha=0.5,
            label=f'{legends[i]}')
        for i in range(len(conditions))
    ]

    # Place custom legend outside the plot
    plt.legend(handles=handles, title="Squared Diff Thresholds", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False,labelspacing=1.5)
    plt.xticks(range(1, 13), month_labels)
    plt.tight_layout()
    plt.show()


if "__main__":
    csv_path = csv_path = "cmip6_era5_data_daily_53.csv"
    # make_variation_heatmap_temperature(csv_path)
    # density_plot_tempereature(csv_path)
    # temperature_trends(csv_path)
    make_monthly_standard_deviation_over_years(csv_path=csv_path)