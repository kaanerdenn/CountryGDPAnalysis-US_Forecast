import numpy as np
import pandas as pd
import warnings
# Importing necessary libraries
import matplotlib.pyplot as plt
import geopandas
import pycountry
import plotly.express as ex
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Suppressing warnings for clean output
warnings.filterwarnings('ignore')

# Setting display options for pandas DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Loading the GDP dataset
df = pd.read_csv("gdp_csv.csv")

# Displaying the first and last few rows of the dataset
print(df.head())
print(df.tail())

# Providing a statistical summary of the dataset
print(df.describe().T)

# Checking for missing values in the dataset
print(df.isnull().sum())

# Aggregating GDP values by country and selecting the top 15 countries
Sum = df['Value'].groupby(df['Country Code']).sum()
first_15 = Sum.sort_values(ascending=True)[:15]

# Plotting the top 15 countries with the lowest GDP
first_15.plot(kind='bar', xlim=10, color='red')
plt.show()

# Listing unique country names in the dataset
print(df["Country Name"].unique())

# Filtering out non-country entities like regions and income groups
countries = [...]
df_country = df.loc[~df['Country Name'].isin(countries)]

# Replacing some country names for consistency with mapping data
df_country = df_country.replace('United States', 'United States of America')

# Creating a line plot of GDP over the years for each country
annotations = []
fig = ex.line(df_country, x="Year", y="Value", color="Country Name", line_group="Country Name", hover_name="Country Name")
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05, xanchor='left', yanchor='bottom', text='GDP over the years (1960 - 2016)', font=dict(family='Arial', size=30), showarrow=False))
fig.update_layout(annotations=annotations)
fig.show()

# Loading the world map for plotting
world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))

# Merging the world map with the GDP data
df_country_final = world.merge(df_country, how="left", left_on=['name'], right_on=['Country Name'])

# Checking for missing values after the merge
print(df_country_final.isnull().sum())

# Plotting a world map with GDP data
df_country_final.plot('Value', figsize=(20, 14), legend=True, legend_kwds={"label": "Gdp By Country", "orientation": "horizontal"})

# Focusing on the United States for time series analysis
gdp = pd.read_csv("gdp_csv.csv")
gdp['Date'] = pd.to_datetime(gdp.Year, format='%Y')
gdp.set_index('Date', inplace=True)
gdp = gdp.loc[gdp["Country Name"] == "United States"]

# Time series modeling using SARIMAX
model = SARIMAX(gdp.Value, order=(3,1,3))
result_AR = model.fit()

