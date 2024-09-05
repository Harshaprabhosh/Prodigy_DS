######### TASK 5 #########

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
import geopandas as gpd
from shapely.geometry import Point
import os
from datetime import datetime

file_path = 'C:/Users/91960/Desktop/python_ws/reduced_US_Accidents_March23.csv'
accident_data = pd.read_csv(file_path)



# Date is in mixed format. So we parse
accident_data['Start_Time'] = accident_data['Start_Time'].apply(lambda x: parser.parse(x, fuzzy=True) if pd.notnull(x) else pd.NaT)
accident_data['End_Time'] = accident_data['End_Time'].apply(lambda x: parser.parse(x, fuzzy=True) if pd.notnull(x) else pd.NaT)


# Converting Start_Time and End_Time to datetime format
accident_data['Start_Time'] = pd.to_datetime(accident_data['Start_Time'])
accident_data['End_Time'] = pd.to_datetime(accident_data['End_Time'])

# Dropping columns with significant missing data
columns_to_drop = ['End_Lat', 'End_Lng', 'Wind_Chill(F)', 'Precipitation(in)', 'Wind_Speed(mph)']
accident_data_cleaned = accident_data.drop(columns=columns_to_drop)

# Dropping rows with missing values
key_columns = ['Street', 'City', 'Zipcode', 'Timezone', 'Weather_Condition', 'Temperature(F)',
               'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction']
accident_data_cleaned = accident_data_cleaned.dropna(subset=key_columns)

# New cleaned dataset
print("Cleaned dataset size:", accident_data_cleaned.shape)



# Extracting hour and day of the week from Start_Time
accident_data_cleaned['Hour'] = accident_data_cleaned['Start_Time'].dt.hour
accident_data_cleaned['Day_of_Week'] = accident_data_cleaned['Start_Time'].dt.day_name()

# Plottiong distribution of accidents by hour
plt.figure(figsize=(10, 6))
sns.countplot(data=accident_data_cleaned, x='Hour', order=range(24), palette='viridis')
plt.title('Distribution of Accidents by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Accidents')
plt.show()

# Plotting the distribution of accidents by day of week
plt.figure(figsize=(10, 6))
sns.countplot(data=accident_data_cleaned, x='Day_of_Week',
              order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
              palette='viridis')
plt.title('Distribution of Accidents by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Accidents')
plt.show()

# Analyzing the impact of weather on accident severity
plt.figure(figsize=(12, 6))
sns.countplot(data=accident_data_cleaned, x='Weather_Condition', hue='Severity',
              palette='coolwarm',
              order=accident_data_cleaned['Weather_Condition'].value_counts().index[:10])
plt.title('Accidents by Weather Condition and Severity')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.legend(title='Severity')
plt.show()



# Creating a GeoDataFrame for accident prone locations
geometry = [Point(xy) for xy in zip(accident_data_cleaned['Start_Lng'], accident_data_cleaned['Start_Lat'])]
geo_df = gpd.GeoDataFrame(accident_data_cleaned, geometry=geometry)

# Plot accident hotspots using a hexbin plot
plt.figure(figsize=(12, 8))
plt.hexbin(accident_data_cleaned['Start_Lng'], accident_data_cleaned['Start_Lat'],
           gridsize=50, cmap='YlOrRd', mincnt=1)
plt.colorbar(label='Number of Accidents')
plt.title('Accident Hotspots')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# Analyzes the impact of contributing factors on severity of accident
contributing_factors = ['Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
                        'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

# Plot contributing factors' impact on severity
contributing_factors_data = accident_data_cleaned[contributing_factors + ['Severity']].melt(id_vars='Severity', value_vars=contributing_factors)

plt.figure(figsize=(12, 8))
sns.countplot(data=contributing_factors_data[contributing_factors_data['value']], x='variable',
              hue='Severity', palette='viridis')
plt.title('Impact of Contributing Factors on Accident Severity')
plt.xlabel('Contributing Factors')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.legend(title='Severity')
plt.show()