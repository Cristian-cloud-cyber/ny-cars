# Write this instruction in your terminal:
# streamlit run "c:..."      # Link to your python File
import os
import zipfile
import csv
import zipfile
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import itertools
import warnings
import streamlit as st
import plotly.express as px
import plotly 
import matplotlib
import altair as alt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
# 1. DATA INGESTION ----------------------------------------------------------------------------------------------------------------

os.chdir(r"Your_Local_Repo_Archive") # Replace with your directory
os.listdir()


# opening the archive for reading
with zipfile.ZipFile('./archive.zip', 'r') as archivio:
    # opening the first csv file within the archive and saving to a DataFrame.
    with archivio.open('Car_Rates.csv') as file1:
        cars_rates = pd.read_csv(io.TextIOWrapper(file1, 'utf-8'))

    # opening the second csv file within the archive and saving to a DataFrame.
    with archivio.open('New_York_cars.csv') as file2:
        ny_cars = pd.read_csv(io.TextIOWrapper(file2, 'utf-8'))

# 1. DATA QUALITY DATA INGESTION -------------------------------------------------------------------------------------------------------------------------------

##  1.a Dataframe Reading for Cars Rate ---------------------------------------------------------------------------------

cars_rates_copy=cars_rates.copy()
cars_rates_copy['Brand'] = cars_rates_copy['Brand'].str.upper()
cars_rates_copy['Model'] = cars_rates_copy['Model'].str.upper()
cars_rates_copy['Model'] = cars_rates_copy['Model'].str.slice(0,-1)
cars_rates_copy['NAME']= cars_rates_copy[['Year', 'Brand', 'Model',]].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
cars_rates_copy = cars_rates_copy[['Car_name', 'NAME'] + [col for col in cars_rates_copy.columns if col not in ['Car_name', 'NAME']]]
# print("Information about Cars Rates Dataframe",cars_rates.info())
# print("Description about Cars Rates Dataframe's quantitative variables",cars_rates.describe().T)

# How many duplicates are there in this dataframe?
# print(f"There are {cars_rates.duplicated(subset='Car_name', keep='first').sum()} duplicated values in this dataframe.")
# print("This info will be useful when we are making merges or unions for our data.")

# Count the number of rows with at least a NaN
# print("total rows in this dataframe: ", cars_rates.shape[0])
# print("rows with at east a NaN: ",cars_rates.isna().any(axis=1).sum())
# print("These are the number of rows with all values filled in: ",cars_rates.shape[0]-cars_rates.isna().any(axis=1).sum())
# print(" '%' of rows without NaNs: ", (cars_rates.shape[0]-cars_rates.isna().any(axis=1).sum())/cars_rates.shape[0]*100,'%')


# 1.b Dataframe Reading for NY cars -------------------------------------------------------------------------------------------------

# print(ny_cars.head(6))
# print(ny_cars[ny_cars['brand'] == 'Land_Rover'])
# print("Information about NY_Cars's Dataframe: ", ny_cars.info())

# print("Description about NY_Cars Dataframe's quantitative variables: ", ny_cars.describe().T)

# Count the number of rows that contain at least one NaN
# print("These are the number of total rows in the dataframe: ", ny_cars.shape[0])
# print("These are the number of rows with at least one NaN: ",ny_cars.isna().any(axis=1).sum())
# print("These are the number of rows with all values filled in: ",ny_cars.shape[0]-ny_cars.isna().any(axis=1).sum())
# print(" '%' of rows without NaNs: ", (ny_cars.shape[0]-ny_cars.isna().any(axis=1).sum())/ny_cars.shape[0]*100,'%')

# 2.a DATA PREPARATION BEFORE OUR DF MERGING ----------------------------------------------------------------------------------------------------------

# Initial Preparations and Editing
ny_cars_copy=ny_cars.copy()
ny_cars_copy['Year'] = ny_cars_copy['Year'].astype(int)
ny_cars_copy['brand'] = ny_cars_copy['brand'].str.replace('_', ' ')
ny_cars_copy=ny_cars_copy.rename(columns={'name':'Car_name'})

#---------------------------------------------------------

# Adjustments for merging's conditions between our 2 dataframes
# It's preferred to get deep into our adjustments in a step-by-step way, to control better our implementations

# 'Year', 'brand', 'Model_1' Conditions
temp_df = ny_cars_copy.copy()
temp_df[['Model_1', 'Model_2']] = ny_cars_copy['Model'].str.split(pat=' ', n=1, expand=True)
temp_df['NAME'] = temp_df[['Year', 'brand', 'Model_1']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).str.upper()
ny_cars_copy['NAME'] = temp_df['NAME']
ny_cars_copy= ny_cars_copy[['Car_name', 'NAME'] + [col for col in ny_cars_copy.columns if col not in ['Car_name', 'NAME']]]

ny_cars_copy['Check'] = ny_cars_copy['NAME'].isin(cars_rates_copy['NAME']).astype(str)
ny_cars_copy['Year'] = ny_cars_copy['Year'].astype(int)
#---------------------------------------------------------

# Adding 'Model_2' to amplify our range for those records haven't been categorized into our last iteration yet
temp_df = ny_cars_copy[ny_cars_copy['Check'] == 'False'].copy()
temp_df[['Model_1', 'Model_2', 'Model_3']] = temp_df['Model'].str.split(pat=' ', n=2, expand=True)
temp_df['NAME'] = temp_df[['Year', 'brand', 'Model_1', 'Model_2']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).str.upper()

ny_cars_copy.update(temp_df)
ny_cars_copy['Check'] = ny_cars_copy['NAME'].isin(cars_rates_copy['NAME']).astype(str)
ny_cars_copy['Year'] = ny_cars_copy['Year'].astype(int)

# These are the values not present within table 12102
# These are the values present 163388
#---------------------------------------------------------

# # Adding 'Model_3' to amplify our range for those records haven't been categorized into our last iteration yet
temp_df = ny_cars_copy[ny_cars_copy['Check'] == 'False'].copy()
temp_df[['Model_1', 'Model_2', 'Model_3', 'Model_4']] = temp_df['Model'].str.split(pat=' ', n=3, expand=True)
temp_df['NAME'] = temp_df[['Year', 'brand', 'Model_1', 'Model_2', 'Model_3']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).str.upper()

ny_cars_copy.update(temp_df)
ny_cars_copy['Check'] = ny_cars_copy['NAME'].isin(cars_rates_copy['NAME']).astype(str)
ny_cars_copy['Year'] = ny_cars_copy['Year'].astype(int)

# These are the values not present within table 11487
# These are the values present 164003
#---------------------------------------------------------

# Adding 'Model_4' to amplify our range for those records haven't been categorized into our last iteration yet
temp_df = ny_cars_copy[ny_cars_copy['Check'] == 'False'].copy()
temp_df[['Model_1', 'Model_2', 'Model_3', 'Model_4','Model_5']] = temp_df['Model'].str.split(pat=' ', n=4, expand=True)
temp_df['NAME'] = temp_df[['Year', 'brand', 'Model_1', 'Model_2', 'Model_3','Model_4']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).str.upper()

ny_cars_copy.update(temp_df)
ny_cars_copy['Check'] = ny_cars_copy['NAME'].isin(cars_rates_copy['NAME']).astype(str)
ny_cars_copy['Year'] = ny_cars_copy['Year'].astype(int)

# These are the values not present within table 11402
# These are the values present 164088
#---------------------------------------------------------

# Removing all 'Model_x' but 'Model_2' to amplify our range for those records haven't been categorized into our last iteration yet, only for Land Rover
temp_df = ny_cars_copy[(ny_cars_copy['Check'] == 'False') & (ny_cars_copy['brand'] == 'Land Rover')].copy()
temp_df[['Model_1', 'Model_2', 'Model_3', 'Model_4','Model_5']] = temp_df['Model'].str.split(pat=' ', n=4, expand=True)
temp_df['NAME'] = temp_df[['Year', 'brand','Model_2']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).str.upper()

ny_cars_copy.update(temp_df)
ny_cars_copy['Check'] = ny_cars_copy['NAME'].isin(cars_rates_copy['NAME']).astype(str)
ny_cars_copy['Year'] = ny_cars_copy['Year'].astype(int)

#---------------------------------------------------------

# Removing all 'Model_x' but 'Model_2' to amplify our range for those records haven't been categorized into our last iteration yet, only for Land Rover
temp_df = ny_cars_copy[(ny_cars_copy['Check'] == 'False') & (ny_cars_copy['brand'] == 'Land Rover')].copy()
temp_df[['Model_1', 'Model_2', 'Model_3', 'Model_4','Model_5']] = temp_df['Model'].str.split(pat=' ', n=4, expand=True)
temp_df['NAME'] = temp_df[['Year', 'brand', 'Model_2', 'Model_3','Model_4']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).str.upper()

ny_cars_copy.update(temp_df)
ny_cars_copy['Check'] = ny_cars_copy['NAME'].isin(cars_rates_copy['NAME']).astype(str)
ny_cars_copy['Year'] = ny_cars_copy['Year'].astype(int)

# These are the values not present within table 8410
# These are the values present 167080
#---------------------------------------------------------

# Removing all 'Model_x' but 'Model_2' and 'Model_3' to amplify our range for those records haven't been categorized into our last iteration yet, only for Land Rover
temp_df = ny_cars_copy[(ny_cars_copy['Check'] == 'False') & (ny_cars_copy['brand'] == 'Land Rover')].copy()
temp_df[['Model_1', 'Model_2', 'Model_3', 'Model_4','Model_5']] = temp_df['Model'].str.split(pat=' ', n=4, expand=True)
temp_df['NAME'] = temp_df[['Year', 'brand', 'Model_2', 'Model_3']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).str.upper()

ny_cars_copy.update(temp_df)
ny_cars_copy['Check'] = ny_cars_copy['NAME'].isin(cars_rates_copy['NAME']).astype(str)
ny_cars_copy['Year'] = ny_cars_copy['Year'].astype(int)

# These are the values not present within table 7503
# These are the values present 167987
#---------------------------------------------------------

# Removing all 'Model_x' but 'Model_1', 'Model_2' and 'Model_3' to amplify our range for those records haven't been categorized into our last iteration yet, only for Mercedes-Benz
temp_df = ny_cars_copy[(ny_cars_copy['Check'] == 'False') & (ny_cars_copy['brand'] == 'Mercedes Benz')].copy()
temp_df['brand'] = temp_df['brand'].replace('Mercedes Benz', 'Mercedes-Benz')
temp_df[['Model_1', 'Model_2', 'Model_3', 'Model_4','Model_5']] = temp_df['Model'].str.split(pat=' ', n=4, expand=True)
temp_df['NAME'] = temp_df[['Year', 'brand','Model_1', 'Model_2', 'Model_3']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).str.upper()

ny_cars_copy.update(temp_df)
ny_cars_copy['Check'] = ny_cars_copy['NAME'].isin(cars_rates_copy['NAME']).astype(str)
ny_cars_copy['Year'] = ny_cars_copy['Year'].astype(int)

# These are the values not present within table 6427
# These are the values present 169063
#---------------------------------------------------------

# Keeping only 'Model_1' to amplify our range for those records haven't been categorized into our last iteration yet, only for Mercedes-Benz
temp_df = ny_cars_copy[(ny_cars_copy['Check'] == 'False') & (ny_cars_copy['brand'] == 'Mercedes-Benz')].copy()
temp_df[['Model_1', 'Model_2', 'Model_3']] = temp_df['Model'].str.split(pat=' ', n=2, expand=True)
temp_df['NAME'] = temp_df[['Year', 'brand','Model_1']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).str.upper()

ny_cars_copy.update(temp_df)
ny_cars_copy['Check'] = ny_cars_copy['NAME'].isin(cars_rates_copy['NAME']).astype(str)
ny_cars_copy['Year'] = ny_cars_copy['Year'].astype(int)

# These are the values not present within table 2412
# These are the values present 173078
#---------------------------------------------------------

# Keeping only 'Model_1' and 'Model_2' to amplify our range for those records haven't been categorized into our last iteration yet, only for Mercedes-Benz
temp_df = ny_cars_copy[(ny_cars_copy['Check'] == 'False') & (ny_cars_copy['brand'] == 'Mercedes-Benz')].copy()
temp_df[['Model_1', 'Model_2', 'Model_3']] = temp_df['Model'].str.split(pat=' ', n=2, expand=True)
temp_df['NAME'] = temp_df[['Year', 'brand','Model_1','Model_2']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).str.upper()

ny_cars_copy.update(temp_df)
ny_cars_copy['Check'] = ny_cars_copy['NAME'].isin(cars_rates_copy['NAME']).astype(str)
ny_cars_copy['Year'] = ny_cars_copy['Year'].astype(int)


# These are the values not present within the table= 10
# These are the values present= 175480
#---------------------------------------------------------

# Keeping only 'Model_1','Model_2','Model_3' and 'Model_4' to amplify our range for those records haven't been categorized into our last iteration yet, only for Mercedes-Benz
temp_df = ny_cars_copy[(ny_cars_copy['Check'] == 'False') & (ny_cars_copy['brand'] == 'Mercedes-Benz')].copy()
temp_df[['Model_1', 'Model_2', 'Model_3','Model_4','Model_5']] = temp_df['Model'].str.split(pat=' ', n=4, expand=True)
temp_df['NAME'] = temp_df[['Year', 'brand','Model_1','Model_2','Model_3','Model_4']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).str.upper()

ny_cars_copy.update(temp_df)
ny_cars_copy['Check'] = ny_cars_copy['NAME'].isin(cars_rates_copy['NAME']).astype(str)
ny_cars_copy['Year'] = ny_cars_copy['Year'].astype(int)

# Print Missing value and Ready ones 
# print("These are the missing values for our Merge= ",ny_cars_copy['Check'].value_counts().get('False'))
# print("Are we ready for Merging? ")
if ny_cars_copy['Check'].value_counts().get('True')==ny_cars_copy.shape[0]:
    print ("YES!")
else:
    print("NOT YET")
    ny_cars_copy[ny_cars_copy['Check'] == 'False']

# Let's Merge our DFs ---------------------------

cars_merged=ny_cars_copy.merge(
    cars_rates_copy[['NAME', 'Num_of_reviews','General_rate', 'Comfort','Interior design', 'Performance', 'Value for the money','Exterior styling', 'Reliability']], 
    on="NAME", how="left")
# Let's remove our Check Column pre-merging
cars_merged = cars_merged.drop('Check', axis=1)


# print("Information about Cars Rates Dataframe",cars_merged.info())
# Let's modify our visualisation of our new Dfs
pd.set_option('display.max_columns', 100)

# print("First rows of our new huge dataframe merged: ",cars_merged.head())

# 2.b DATA PREPARATION AFTER OUR DF MERGING ------------------------------------------------------------------

# Let's standardize "Accidents or damage" Field
cars_merged['Accidents or damage'].unique()
# We have only to fill NaN with None Reported
cars_merged['Accidents or damage']=cars_merged['Accidents or damage'].fillna('None Reported')
cars_merged['Accidents or damage'].unique()

#---------------------------------------------------------
# Now it's the turn of New and Used

# Let's see what kind of vaues we have in new&used
# print('Categories in new&used: ',cars_merged['new&used'].unique())

# Since we have other categories (new and used apart), we need to split these different values in a different column to mean that if a car has already used, 
# it could have a certification or not, 
cars_merged['Certified_Used'] = cars_merged['new&used'].apply(lambda x: x if x not in ['New', 'Used'] else np.nan)
# # print(cars_merged['Certified_Used'].unique())

# replacing them with "Used" in new&used column
cars_merged['new&used'] = cars_merged['new&used'].apply(lambda x: x if x in ['New', 'Used'] else 'Used')
# # print(cars_merged['new&used'].unique())

#---------------------------------------------------------
# Clean title's Turn

# First step:
# print(cars_merged['Clean title'].unique())
cars_merged['Clean title'] = cars_merged['Clean title'].apply(lambda x: x if x in ['Yes', 'No'] else np.nan)
# Step to Repeat:
# print(cars_merged['Clean title'].unique())

#---------------------------------------------------------
# Fuel Type

# print('Categories in Fuel Type: ',cars_merged['Fuel type'].unique())

# Let's define categories 
gasoline = ['Gasoline','Regular Unleaded ', 'Premium Unleaded ','Gasolin']
hybrid = ['Hybrid', 'Gasoline/Mild Electric Hybrid ','Plug-In Hybrid ','Plug-In Electric/Gas ']
diesel = ['Diesel','Biodiesel']
other = ['Hydrogen','Compressed Natural Gas ','E85 Flex', 'Xib','Gas']
Unknown = [np.nan, 'Unspecified','Other']

# Let's create a function to categorize our fuel types
def categorize_fuel_type(value):
    if value in gasoline:
        return 'Gasoline'
    elif value in hybrid:
        return 'Hybrid'
    elif value in diesel:
        return 'Diesel'
    elif value in other:
        return 'Other'
    elif value in Unknown:
        return 'Unknown'
    else:
        return value

# Let's apply this new categories without losing original data
cars_merged['Fuel_type_categorized'] = cars_merged['Fuel type'].apply(categorize_fuel_type)

# # print(cars_merged['Fuel_type_categorized'].unique())
# cars_merged['Clean title'] = cars_merged['Clean title'].apply(lambda x: x if x in ['Yes', 'No'] else 'NaN')
# # print(cars_merged['Clean title'].unique())

# Let's place our columns as we want
cars_merged= cars_merged[['Car_name','NAME','new&used','Certified_Used','money','Exterior color','Interior color','Drivetrain','MPG','Fuel_type_categorized','Fuel type'] + [col for col in cars_merged.columns if col not in ['Car_name','NAME','new&used','Certified_Used','money','Exterior color','Interior color','Drivetrain','MPG','Fuel_type_categorized','Fuel type']]]

#---------------------------------------------------------
# Transmission Type's Turn

# # print(np.sort(cars_merged['Transmission'].unique().astype(str)))
# Let's convert all nan texts in nan as a "non-value"

cars_merged = cars_merged.replace(['null', 'Null', 'Nan', 'NaN', 'nan'], np.nan)

# We convert all data in Transmission to split in a few of categories our informations
def categorize_transmission(val):
    # We manage our nan under a label called 'Unknown'
    if pd.isna(val):
        return 'Unknown'

    val = str(val)
    
    # Let's convert in lower string to not have issues with categories' names.
    val = val.lower()
    
    if any(x in val for x in ['cvt','continuously variable','multitronic','single speed','xtronic stepless gear','ivt']):
        return 'Cvt'
    elif any(x in val for x in ['automatic','atomatic', 'a/t', 'auto','steptronic','a',
                                'shiftronic', 'tiptronic','variable','speed dual-clutch',
                                'speed dual clutch','speed double clutch','speed m double clutch',
                                'speed s tronic', '6-spd selectshift trans', 
                                'transmission w/dual shift mode', '6-speed ecoshift dual clutch', 
                                'speed ecoshift dual clutch (dct)', 'ecoshift dual clutch', 
                                'dct', 'sportronic with paddle-shifters','pdk','speed double-clutch','eflite','6ecti','8ecti']) and 'cvt' not in val and 'manual' not in val:
        return 'Automatic'
    elif any(x in val for x in ['manual', 'm/t','m']):
        return 'Manual'
    else:
        return 'Unknown'

# Put our data in a new column
cars_merged['Categorized_Transmission'] = cars_merged['Transmission'].apply(categorize_transmission)


# Let's place our data in a different columns to mantain original data
cars_merged= cars_merged[['Car_name','NAME','new&used','Certified_Used','money','Exterior color','Interior color','Drivetrain','MPG','Fuel_type_categorized','Fuel type','Categorized_Transmission','Transmission'] + [col for col in cars_merged.columns if col not in ['Car_name','NAME','new&used','Certified_Used','money','Exterior color','Interior color','Drivetrain','MPG','Fuel_type_categorized','Fuel type','Categorized_Transmission','Transmission']]]


# This is my code that I used to gain informations about what kind of different transmission exists and in which kind of macro categories I can split them down.
 
# # print(cars_merged.columns)
# # print(cars_merged[cars_merged['Categorized_Transmission']=='Cvt'].shape[0]+cars_merged[cars_merged['Categorized_Transmission']=='Automatic'].shape[0]+cars_merged[cars_merged['Categorized_Transmission']=='Manual'].shape[0])
# # print(
#     cars_merged.shape[0]-(cars_merged[cars_merged['Categorized_Transmission']=='Cvt'].shape[0]
#                           +cars_merged[cars_merged['Categorized_Transmission']=='Automatic'].shape[0]
#                           +cars_merged[cars_merged['Categorized_Transmission']=='Manual'].shape[0]))


# cars_merged[cars_merged['Categorized_Transmission']=='Unknown']['Transmission'].unique()
#---------------------------------------------------------
# MilesPerGallon Type' Turn

cars_merged['MPG'] = cars_merged['MPG'].astype(str)
# Among our data there are values that could be filled by other data, as we know, in the same category, with mode statistic.
def fillna_with_mode(x):
    # Let's compute mode
    mode = x.mode()
    # If our series isn't empty, get me the first element, otherwise get me 'Unknown'
    return mode.iloc[0] if not mode.empty else 'Unknown'

mode = cars_merged.groupby('NAME')['MPG'].transform(fillna_with_mode)

# Let's replace our nan with mode
cars_merged['MPG'] = cars_merged['MPG'].fillna(mode)
cars_merged['MPG']=cars_merged['MPG'].astype(str)
# I got that to cover our values we need to modify our data, this time I prefer to update directly original data
# but it's only to show how I can do it. In real life, I prefer to do as I did before: creating a new different column in which gaining my new modified data.
# Let's give a standard format to our data
cars_merged['MPG'] = cars_merged['MPG'].replace(['nan'], 'Unknown')
cars_merged['MPG'] = cars_merged['MPG'].str.replace('–', '-')
cars_merged['MPG'] = cars_merged['MPG'].str.replace(' ', '')
cars_merged['MPG'] = cars_merged['MPG'].apply(lambda x: '0-' + x if '-' not in x and 'Unknown' not in x else x)
cars_merged['MPG'] = cars_merged['MPG'].replace(['0-0', '0-0.0'], 'Unknown')
cars_merged['MPG'] = cars_merged['MPG'].str.replace('.', '')
def swap_values(x):
    parts = x.split('-')
    
    if len(parts) == 2 and int(parts[0]) > int(parts[1]):
        return parts[1] + '-' + parts[0]
    
    return x
cars_merged['MPG'] = cars_merged['MPG'].apply(swap_values)

# Let's adjust our data to put them in order and to get them respecting the rule as format and limits
def check_right_value(x):
    parts = x.split('-')
    if len(parts) == 2 and int(parts[1]) > 130 and int(parts[1]) < 1000:
        return True
    return False

# Let's apply the rule
# print(cars_merged[cars_merged['MPG'].apply(check_right_value)]['MPG'])

# Let's adjust our ending data by mistakes and added error
cars_merged['MPG'] = cars_merged['MPG'].str.replace('0-255', '0-25')

# # print(cars_merged['MPG'].unique())
def replace_values(x):
    # Split the string into two parts using the symbol '-'
    parts = x.split('-')
    # If the value on the right is less than 1000, return 'Unknown'
    if len(parts) == 2 and int(parts[1]) > 1000:
        return 'Unknown'
    # Otherwise, return the original string
    return x

# Applies the function to each value in the 'MPG' column
cars_merged['MPG'] = cars_merged['MPG'].apply(replace_values)

# Finally let's create a new column in wich we are going to insert the average value of MPG, we will use it to gain information for studying an ecological asset in our data
def calculate_average(x):
    # If the value is 'Unknown', return 'Unknown'
    if x == 'Unknown':
        return 'Unknown'
    # Otherwise, calculate the average of the values to the left and right of the '-' symbol
    else:
        parts = x.split('-')
        average = int((float(parts[0]) + float(parts[1])) / 2)
        return average

# Create a new column 'MPG_Puntual' by applying the function to each value in column 'MPG'
cars_merged['MPG_Puntual'] = cars_merged['MPG'].apply(calculate_average)
cars=cars_merged.copy()
# 3. GRAPHS FOR OUR DATA


# Streamlit App Title -------------------------------------------------------------------------------------------------------------
st.title('Cars in NY')

# Data Display
st.subheader('Data Elaboration Process and Analys')
st.write("To study all the available information we need to analyse all the data we have, so let's look inside our file to answer the question: \n")
st.info("How many dataframes do we have in this file?")

# Open the zip file
with zipfile.ZipFile('archive.zip') as zip:
    # Get a list of the names of the files contained in the zip file
    file_names = zip.namelist()
    st.warning(len(file_names))
    st.info("What are the names of these dataframes?")
    # Display the file names
    st.warning(', '.join(file_names))

    
st.write("Due to get a good analysis we have to see what kind of data we got and try to determine if we can unify all those informations in a unique dataframe. \n Hence, we need to analyse if in our dataframes are there particular informations we can use, to make a unique huge dataframe")
st.write("Let's call inside our study those dataframes")
st.info(f"Let's see the 1° File: {file_names[0]}")
st.write(cars_rates.head(20))

st.info(f"Let's see the 2° File: {file_names[1]}")
st.write(ny_cars.head(20))
st.write("As we can see, we can notice that the main column, to be used to get our dataframe united in only one, refers to car's year-brand names-model. \n In advance, we must clean and prepare our dataframe:")
st.markdown("""
            - For Cars Rates:
                - Let's Create the Column for our Unification, removing the point in the final part of string for each value.
            - For Ny Cars:
                - Let's create the same columns from the data coined in this dataframe with the same characteristics of the "car's rates"' one
            """)
st.write("Once we did it, let's go for our Unification")
st.info("We're seeing all data could be unified in a unique dataframe, because in Cars Rates we have:")
st.warning(f"{cars_rates.duplicated(subset='Car_name', keep='first').sum()} duplicates")
st.info("So all data inside **Cars Rates** could be connected to **Ny Cars**, using cars names as primary key, after getting dataframes cleaned and prepared for it")
st.write("We will stardardize and clean these fields:")
st.markdown("""
            - Accident & Damage, to put under 'NaN' those values seen asa 'None Reported',
            - New and Used to put all those "Certified" Values in a different column, and denote their position in the New and Used column as Used,
            - Fuel Type as only 4 type of Values as Gasoline, Diesel, Hybrid and Other, or Unknown as we can't get then another label from before,
            - Transmission type for visualize which car is or Automatic, or Cvt, or Manual or Unknown types, and putting our new categories in Transmission field,
            - MPG_Puntual, created to host all data contained in MPG field, cleaned and corrected.            
            """)
st.warning("Now we have to check how our fields and different variables has to be prepared to our Analysis.")

st.write(cars.head(20))

#-----------------------------------------------------------------------------------------------------------------------------------------

# PIE CHART --------------------------------------------------------------------------------------
# Create the data
# Create the figure
# Display the figure in Streamlit
st.info("Now we are going to get a graphic and interesting view of the data we have:")
st.plotly_chart(px.pie(values=cars['new&used'].value_counts().values, 
             names=cars['new&used'].value_counts().index, 
             title="Partition of Cars splitted by Used and New", 
             color_discrete_sequence=['orange', 'blue']), use_container_width=True)
st.info("It's important to underline this aspect, because in New York only a few part of cars are new, so we expect that it could be a great difference in price, but also in performance.")
st.warning("Let's go inside our date to analyse these differences between Categories as New and Used")
# # Create a pie chart
# piechart, ax = plt.subplots(figsize=(2, 2))
# ax.pie( cars['new&used'].value_counts(), labels=cars['new&used'].value_counts().index, explode=(0.1,0),autopct='%1.1f%%',shadow=True ,startangle=90)
# ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# # Display the chart in Streamlit
# st.pyplot(piechart)

# BOXPLOT --------------------------------------------------------------------------------

# box, axes = plt.subplots(1, 2, figsize=(12, 6))
# # Create the boxplot for used cars
# ny_cars[ny_cars['new&used'] == 'Used']['money'].plot(kind='box', ax=axes[0])
# axes[0].set_ylim(0, 120000)
# axes[0].set_title('Used Cars')
# axes[0].set_ylabel('Money')
# # Create the boxplot for new cars
# ny_cars[ny_cars['new&used'] == 'New']['money'].plot(kind='box', ax=axes[1])
# axes[1].set_ylim(0, 120000)
# axes[1].set_title('New Cars')
# axes[1].set_ylabel('Money')

# # Adjust layout and display the chart in Streamlit
# plt.tight_layout()
# st.pyplot(box)

# Create the data
new_cars = cars[cars['new&used'] == 'New']['money']
used_cars = cars[cars['new&used'] == 'Used']['money']

# Create the figure
fig = go.Figure()

# Add box plots
fig.add_trace(go.Box(y=new_cars, name='New', marker_color='blue'))
fig.add_trace(go.Box(y=used_cars, name='Used', marker_color='orange'))
fig.update_layout(title="Prices of Cars divided by New and Used", yaxis=dict(range=[0, 120000], autorange=False))
# Display the figure in Streamlit
st.plotly_chart(fig, use_container_width=True)


st.warning("Let's see also which is the top brand with high prices in this market:")
# BAR PLOT FILTERED AND SORTED FOR 2 DIFFERENT VARIABLES ---------------------------------------------

# This graph represents the car's ranking in order for sums of money invested to gain cars distincted by brand,
# while bars shows the average value to pay in dolars to purchase that brand divided by new and used cars.
# BAR PLOT ORIZZONTALE IN ORDINE PER BRAND
# Create the data
data = cars[cars['brand'] != 'Unknown'].groupby(['brand', 'new&used'])['money'].mean().reset_index()
# Sort the brands based on the average value of 'money'
brands_order = data.groupby('brand')['money'].mean().sort_values(ascending=False).index.tolist()
# Create the figure
# Aggiungi un widget slider per selezionare il numero di record da visualizzare
n = st.number_input('How many record of brands do you want to visualize?', min_value=1, max_value=len(brands_order), value=10)

# Filtra i primi n brand
top_brands = brands_order[:n]

# Filtra i dati per includere solo i primi n brand
filtered_data = data[data['brand'].isin(top_brands)]

# Crea il grafico con i dati filtrati
fig = px.bar(filtered_data, y='brand', x='money', color='new&used', 
             category_orders={'brand': top_brands},
             labels={'money': 'Money', 'brand': 'Brand'},
             title='Mean of Money for New and Used Cars for Each Brand',
             barmode='group',
             color_discrete_map={'Used':'orange', 'New':'blue'})
fig.add_annotation(text='In Descending Order of Average Price in Cars',
                   xref='paper', yref='paper', x=1, y=0.05,
                   showarrow=False, font=dict(size=12))

# Mostra il grafico in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Histogramma distributivo con Kernel non Normalizzato -----------------------------------

import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde

# # Create the data
# used_cars = ny_cars[ny_cars['new&used'] == 'Used']['money']
# new_cars = ny_cars[ny_cars['new&used'] == 'New']['money']

# # Create the figure
# fig = go.Figure()

# # Add histograms to the figure
# fig.add_trace(go.Histogram(x=used_cars, nbinsx=5000, name='Used', marker_color='orange', histnorm=''))  # Added histnorm=''
# fig.add_trace(go.Histogram(x=new_cars, nbinsx=5000, name='New', opacity=0.50, histnorm=''))  # Added histnorm=''

# # Calculate KDE for used cars
# kde_used = gaussian_kde(used_cars)
# x_range_used = np.linspace(min(used_cars), max(used_cars), 5000)
# y_range_used = kde_used(x_range_used) * len(used_cars) * (max(used_cars) - min(used_cars)) / 5000

# # Calculate KDE for new cars
# kde_new = gaussian_kde(new_cars)
# x_range_new = np.linspace(min(new_cars), max(new_cars), 5000)
# y_range_new = kde_new(x_range_new) * len(new_cars) * (max(new_cars) - min(new_cars)) / 5000

# # Add KDE lines to the figure
# fig.add_trace(go.Scatter(x=x_range_used, y=y_range_used, mode='lines', name='Used KDE', line=dict(color='orange')))
# fig.add_trace(go.Scatter(x=x_range_new, y=y_range_new, mode='lines', name='New KDE', line=dict(color='blue')))

# # Overlay both histograms
# fig.update_layout(barmode='overlay')

# # Set the axis limits
# fig.update_layout(
#     autosize=False,
#     width=1200,
#     height=600,
#     xaxis=dict(range=[0, 100000], autorange=False),
#     yaxis=dict(range=[0, 4800], autorange=False),
# )

# # Set the title and axis labels
# fig.update_layout(title_text='Counts distribution to compare Category New & Used', xaxis_title='Money', yaxis_title='Counts')

# # Display the plot in Streamlit
# st.plotly_chart(fig)

#-------------------------------------------------------------------
# Let's compare Used and New cars for Money, we are using Kernel ditribution
st.warning("Now, we want to see these data from another prospective, but to do that we need to see them introducing a Kernel distribution graph that can normalize data as we neee to compare different distributions.")

used_cars = ny_cars[ny_cars['new&used'] == 'Used']['money']
new_cars = ny_cars[ny_cars['new&used'] == 'New']['money']
group_labels = ['Used', 'New']
fig = ff.create_distplot([used_cars, new_cars], group_labels, bin_size=500, show_rug=False, colors=['orange','blue'])
fig.update_layout(
    autosize=False,
    width=1200,
    height=600,
    xaxis=dict(range=[0, 140000], autorange=False),
    yaxis=dict(range=[0, 0.000060], autorange=False),
)

fig.update_layout(title_text='KDE and Histograms', xaxis_title='Money')
st.plotly_chart(fig, use_container_width=True)
st.info("So we can say that we a great average difference between these category in price.")


# HISTORICAL LINEPLOT FOR EACH BRAND BY NEW AND USED --------------------------------------------------

import streamlit as st
import plotly.graph_objects as go
st.warning("Let's see if between these categories there are difference for Performance:")
# Filter Button 
brands = np.append(cars['brand'].unique(), 'All')
filter_brand = st.selectbox('Select your preferred Brand:', brands)

if filter_brand == 'All':
    data = cars[cars['Fuel_type_categorized'] != 'Unknown'].groupby(['Year', 'new&used'])['Performance'].mean().reset_index()
else:
    data = cars[(cars['Fuel_type_categorized'] != 'Unknown') & (cars['brand'] == filter_brand)].groupby(['Year', 'new&used'])['Performance'].mean().reset_index()

fig = go.Figure()
colors = {'Used': 'orange', 'New': 'blue'}
for new_used in data['new&used'].unique():
    subset = data[data['new&used'] == new_used]
    fig.add_trace(go.Scatter(x=subset['Year'], y=subset['Performance'], mode='lines+markers', name=new_used, line=dict(color=colors[new_used])))
fig.update_layout(xaxis_title='Year', yaxis_title='Average Performance', title='Line Plot of cars')
st.plotly_chart(fig, use_container_width=True)
st.info("Let's say that our data are quite weird in 2013, as we can describe it as an outlier.")

# GAUSSIAN HIST/LINE PLOT -------------------------------------------------------------------------

# Crea la figura
# plt.figure(figsize=(12, 6))
# # Crea gli istogrammi
# sns.histplot(ny_cars[ny_cars['new&used'] == 'Used']['money'], kde=True, color='orange', label='Used')
# sns.histplot(ny_cars[ny_cars['new&used'] == 'New']['money'], kde=True, color='blue', label='New', alpha=0.3)
# # Imposta i limiti degli assi
# plt.xlim(0, 140000)
# plt.ylim(0, 5000)
# # Imposta il titolo e le etichette degli assi
# plt.title('Cars')
# plt.xlabel('Money')
# # Mostra la legenda
# plt.legend()
# # Regola il layout
# plt.tight_layout()
# # Visualizza il grafico in Streamlit
# st.pyplot(plt.gcf())



# BAR PLOT FOR COMPARISON IN FUEL TYPES--------------------------------------------------------------------------------
import plotly.express as px
st.write("Pass over on another important topic, about:")
st.subheader("The behaviour of the market respect Fuel Types")
st.warning("How are distrubuited Cars in New York for Fuel Types?")
data = cars[cars['Fuel_type_categorized'] != 'Unknown'].groupby(['Fuel_type_categorized', 'new&used']).size().reset_index(name='count')
fig = px.bar(data, y='Fuel_type_categorized', x='count', color='new&used', 
             category_orders={'Fuel_type_categorized': data.groupby('Fuel_type_categorized')['count'].sum().sort_values(ascending=False).index.tolist()},
             labels={'count': 'Number of Cars', 'Fuel_type_categorized': 'Fuel Type Categorized'},
             title='Number of Cars for Each Fuel Type',
             barmode='group',
             color_discrete_map={'Used':'orange', 'New':'blue'})
st.plotly_chart(fig, use_container_width=True)
st.info("We can notice how Gasoline is the most preferred Fuel despite to others.")
#----------------------------------------------------------------------------------------------------------------
# import plotly.express as px

# data = cars[cars['Fuel_type_categorized'] != 'Unknown'].groupby(['Fuel_type_categorized', 'new&used'])['money'].sum().reset_index()
# fig = px.bar(data, y='Fuel_type_categorized', x='money', color='new&used', 
#              category_orders={'Fuel_type_categorized': data.groupby('Fuel_type_categorized')['money'].sum().sort_values(ascending=False).index.tolist()},
#              labels={'money': 'Money in Mld', 'Fuel_type_categorized': 'Fuel Type Categorized'},
#              title='Sum of Money for New and Used Cars for Each Fuel Type',
#              barmode='group',
#              color_discrete_map={'Used':'orange', 'New':'blue'})
# st.plotly_chart(fig, use_container_width=True)

#-------------------------------------------------------------------------------------------------------------------

# # Order 'Fuel_type_categorized' based on the sum of 'money'
# plt.figure(figsize=(10, 4))
# # Add a grid with only vertical lines to the background
# plt.grid(True, which='both', color='grey', linewidth=0.3, linestyle='--', zorder=0, axis='x')
# sns.barplot(x='money', y='Fuel_type_categorized', hue='new&used', 
#             data=cars[cars['Fuel_type_categorized'] != 'Unknown'].groupby(['Fuel_type_categorized', 'new&used'])['money'].sum().reset_index(), 
#             errorbar=None, 
#             order=cars[cars['Fuel_type_categorized'] != 'Unknown'].groupby(['Fuel_type_categorized', 'new&used'])['money'].sum().reset_index().groupby('Fuel_type_categorized')['money'].sum().sort_values(ascending=False).index, zorder=3)
# plt.xlabel('Money in Mld')
# plt.ylabel('Fuel Type Categorized')
# plt.title('Sum of Money for New and Used Cars for Each Fuel Type')
# plt.legend(title='New & Used')
# plt.box(False)
# st.pyplot(plt.gcf())





# ------------------------------------------------------------------------------------------------

# Order 'brand' based on the Mean of 'money'
# plt.figure(figsize=(10, 6))
# # Add a grid with only vertical lines to the background
# plt.grid(True, which='both', color='grey', linewidth=0.3, linestyle='--', zorder=0, axis='x')
# # I would like to see the average value of maney that people pay for a brand, in a order that represents the value of brand in our DF for each brand on the total
# sns.barplot(x='money', y='brand', hue='new&used', 
#             data=cars[cars['brand'] != 'Unknown'].groupby(['brand', 'new&used'])['money'].mean().reset_index(), 
#             errorbar=None, 
#             order=cars[cars['brand'] != 'Unknown'].groupby(['brand', 'new&used'])['money'].sum().reset_index().groupby('brand')['money'].sum().sort_values(ascending=False).index
# , zorder=3)
# plt.xlabel('Money in Mld')
# plt.ylabel('Brand')
# plt.title('Mean of Money for New and Used Cars for Each Brand')
# plt.legend(title='New & Used')
# plt.ylim(6.5)
# plt.box(False)
# st.pyplot(plt.gcf())

# LINE PLOT/HISTORICAL SIRIES -------------------------------------------------------------------------

# This Graph represents how many cars there are along years, distincted by Fuel Type
# To gain how much the car's market are moving towards other ecological fuel sources in this particular sector.
# Modify this list to suit your needs


st.warning("Now we want to study how this fuel market for cars evolved along years:")
filter_value = st.selectbox('Select car type:', ('Used', 'New', 'Both'))
# If 'Both' is selected, use both 'Used' and 'New'
if filter_value == 'Both':
    filter_value = ['Used', 'New']
else:
    filter_value = [filter_value]  # Make sure filter_value is always a list

# -----------------------------------------------------------------------------------------------------

# Create the data
    
data = cars[(cars['Fuel_type_categorized'] != 'Unknown') & (cars['new&used'].isin(filter_value))].groupby(['Year', 'Fuel_type_categorized'])['new&used'].count().reset_index()
# Create the figure
fig = go.Figure()
# Add lines to the chart
for fuel_type in data['Fuel_type_categorized'].unique():
    subset = data[data['Fuel_type_categorized'] == fuel_type]
    fig.add_trace(go.Scatter(x=subset['Year'], y=subset['new&used'], mode='lines', name=fuel_type))
# Set the axis labels and the title
fig.update_layout(xaxis_title='Year', yaxis_title='Count of new&used', title='Line Plot for Cars divided by Fuel categories')
# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)
st.info("As we can see the cars market didn't want to refuse to increase neither after the hole provoqued by Covid-Sars-19. We are going to comparise so our fuel types cars along another kind of variable for measuring their performance in **Average MPG**.")



# filter_brand = ['Toyota']  # Inserisci i marchi che vuoi filtrare
# plt.figure(figsize=(10, 6))
# plt.grid(True, which='both', color='grey', linewidth=0.3, linestyle='--', zorder=0, axis='y')
# min_year = cars['Year'].min()
# max_year = cars['Year'].max()
# for new_used in cars[cars['brand'].isin(filter_brand)].groupby('new&used')['Performance'].mean().reset_index()['new&used'].unique():
#     subset = cars[cars['brand'].isin(filter_brand)].groupby(['Year', 'new&used'])['Performance'].mean().reset_index()[cars[cars['brand'].isin(filter_brand)].groupby(['Year', 'new&used'])['Performance'].mean().reset_index()['new&used'] == new_used]
#     # I check if there is enough data to draw the line.
#     if len(subset) > 1:
#         plt.plot(subset['Year'], subset['Performance'], label=new_used)
#     else:
#         plt.scatter(subset['Year'], subset['Performance'], label=new_used)
#         min_year = min(min_year, subset['Year'].min())
#         max_year = max(max_year, subset['Year'].max())
# plt.xlim(min_year, max_year)
# plt.xlabel('Year')
# plt.ylabel('Average Performance')
# plt.legend(title='New & Used')
# plt.title('Line Plot of cars')
# plt.box(False)
# st.pyplot(plt.gcf())

# HISTORICAL LINEPLOT FOR AVERAGE MILES PER GALLON BY FUEL TIPES --------------------------------------------------

# # We can show a graph to show which is the hystorical siries of Mileage for Gallon during years distincted by fuel type, to see of there are some improvements by car's market
# cars['MPG_Puntual'] = pd.to_numeric(cars['MPG_Puntual'], errors='coerce')
# # Crea il plot
# plt.figure(figsize=(10, 6))
# # Aggiungi una griglia con solo linee verticali sullo sfondo
# plt.grid(True, which='both', color='grey', linewidth=0.3, linestyle='--', zorder=0, axis='y')
# for fuel_type in cars[cars['Fuel_type_categorized'] != 'Unknown'].groupby(['Year', 'Fuel_type_categorized'])['MPG_Puntual'].mean().reset_index()['Fuel_type_categorized'].unique():
#     subset = cars[cars['Fuel_type_categorized'] != 'Unknown'].groupby(['Year', 'Fuel_type_categorized'])['MPG_Puntual'].mean().reset_index()[cars[cars['Fuel_type_categorized'] != 'Unknown'].groupby(['Year', 'Fuel_type_categorized'])['MPG_Puntual'].mean().reset_index()['Fuel_type_categorized'] == fuel_type]
#     plt.plot(subset['Year'], subset['MPG_Puntual'], label=fuel_type)
# plt.xlabel('Year')
# plt.ylabel('Average MPG_Puntual')
# plt.legend(title='Fuel Type')
# plt.title('Line Plot of cars')
# plt.box(False)
# st.pyplot(plt.gcf())

import plotly.graph_objects as go

# Crea un menu a discesa per la selezione
selected_type = st.selectbox("Let's select if you want to visualize Used or New cars... or both", ['Both','Used', 'New'])

if selected_type == 'Both':
    cars_filtered = cars
else:
    cars_filtered = cars[cars['new&used'] == selected_type]

cars_filtered['MPG_Puntual'] = pd.to_numeric(cars_filtered['MPG_Puntual'], errors='coerce')
fig = go.Figure()
for fuel_type in cars_filtered[cars_filtered['Fuel_type_categorized'] != 'Unknown']['Fuel_type_categorized'].unique():
    subset = cars_filtered[cars_filtered['Fuel_type_categorized'] == fuel_type].groupby(['Year', 'Fuel_type_categorized'])['MPG_Puntual'].mean().reset_index()
    fig.add_trace(go.Scatter(x=subset['Year'], y=subset['MPG_Puntual'], mode='lines', name=fuel_type))
fig.update_layout(
    title='Line Plot of ' + selected_type + ' Cars',
    xaxis_title='Year',
    yaxis_title='Average MPG_Puntual',
    legend_title='Fuel Type'
)
st.plotly_chart(fig, use_container_width=True)
st.info("We can notice a loss of information relatives the latest evolution of our hystorical series. But we can assume that more data are available for each category, more we can analyse about our market, so for New Cars we can see some holes in information, despite for Used, positively.")



# 4. FITTING MODELS TO OUR DATA ----------------------------------------------------------------------------------------------------------

warnings.filterwarnings('ignore')

#-----------------------------------------------------------------------------------------------------------------------------------
# # Definisci i valori per il filtro
# filter_value = ['Used']
# years_to_predict = 2 
# # Calcola il totale delle auto per anno
# if not filter_value:
#     subset = cars.groupby('Year')['new&used'].count().reset_index()
# else:
#     subset = cars[cars['new&used'].isin(filter_value)].groupby('Year')['new&used'].count().reset_index()
# # Definisci i range per p, d, q
# p = d = q = range(0, 3)
# # Genera tutte le diverse combinazioni di p, d, q
# pdq = list(itertools.product(p, d, q))
# # Trova l'ordine ottimale
# best_aic = np.inf
# best_pdq = None
# for order in pdq:
#     try:
#         model = ARIMA(subset['new&used'], order=order)
#         results = model.fit()     
#         # Aggiorna l'ordine ottimale se l'AIC corrente è inferiore al migliore finora
#         if results.aic < best_aic:
#             best_aic = results.aic
#             best_pdq = order
#     except:
#         continue
# # Crea il modello ARIMA con l'ordine ottimale
# model = ARIMA(subset['new&used'], order=best_pdq)
# results = model.fit()
# # Fai le previsioni
# arima_predictions = results.predict(start=0, end=len(subset) - 1)
# # Crea il modello di regressione lineare
# linear_model = LinearRegression()
# linear_model.fit(subset[['Year']], subset['new&used'])
# # Calcola le previsioni del modello lineare
# linear_predictions = linear_model.predict(subset[['Year']])
# # Calcola l'errore quadratico medio (MSE) per il modello lineare
# linear_mse = np.mean((subset['new&used'] - linear_predictions)**2)
# # Calcola l'errore quadratico medio (MSE) per il modello ARIMA
# arima_mse = np.mean((subset['new&used'] - arima_predictions)**2)
# # Scegli il modello con l'MSE più basso
# if linear_mse < arima_mse:
#     predictions = linear_predictions
# else:
#     predictions = arima_predictions
# # Numero di anni da prevedere
#  # Modifica questo valore con il numero di anni che desideri prevedere
# # Estendi l'array degli anni per includere gli anni futuri
# future_years = np.array(range(max(subset['Year']) + 1, max(subset['Year']) + years_to_predict + 1)).reshape(-1, 1)
# # Calcola le previsioni per gli anni futuri
# future_predictions = linear_model.predict(future_years)
# # Unisci le previsioni attuali e future
# all_years = np.concatenate((subset[['Year']].values, future_years))
# all_predictions = np.concatenate((predictions, future_predictions))
# # Crea il plot
# plt.figure(figsize=(10, 6))
# # Aggiungi una griglia con solo linee verticali sullo sfondo
# plt.grid(True, which='both', color='grey', linewidth=0.3, linestyle='--', zorder=0, axis='y')
# # Crea il grafico dei dati reali
# plt.plot(subset['Year'], subset['new&used'], label='Actual')
# # Plotta le previsioni
# plt.plot(all_years[1:], all_predictions[1:], color='red', linestyle='--', label='Predicted')
# plt.xlabel('Year')
# plt.ylabel('Count of new&used')
# plt.title('Line Plot of cars')
# plt.legend()
# plt.box(False)
# st.pyplot(plt.gcf())
st.warning("In the next graph, I would show how to get the right optimized model to different historical line series")
# --------------------------------------------------------------------------------------------------------------------

import plotly.graph_objects as go

filter_options = st.multiselect('Select filter values', ['Used', 'New'], default=['Used', 'New'])
years_to_predict = 1
# Calculate the total number of cars per year
if not filter_options:
    subset = cars.groupby('Year')['new&used'].count().reset_index()
else:
    subset = cars[cars['new&used'].isin(filter_options)].groupby('Year')['new&used'].count().reset_index()
# Define the ranges for p, d, q
p = d = q = range(0, 3)
# Generate all different combinations of p, d, q
pdq = list(itertools.product(p, d, q))

# Find the optimal order
best_aic = np.inf
best_pdq = None
for order in pdq:
    try:
        model = ARIMA(subset['new&used'], order=order)
        results = model.fit()
        
        # Update the optimal order if the current AIC is lower than the best so far
        if results.aic < best_aic:
            best_aic = results.aic
            best_pdq = order
    except:
        continue

# Create the ARIMA model with the optimal order
model = ARIMA(subset['new&used'], order=best_pdq)
results = model.fit()

# Make predictions
arima_predictions = results.predict(start=0, end=len(subset) - 1)

# Create the linear regression model
linear_model = LinearRegression()
linear_model.fit(subset[['Year']], subset['new&used'])

# Calculate the linear model predictions
linear_predictions = linear_model.predict(subset[['Year']])

# Calculate the mean squared error (MSE) for the linear model
linear_mse = np.mean((subset['new&used'] - linear_predictions)**2)

# Calculate the mean squared error (MSE) for the ARIMA model
arima_mse = np.mean((subset['new&used'] - arima_predictions)**2)

# Choose the model with the lowest MSE
if linear_mse < arima_mse:
    predictions = linear_predictions
else:
    predictions = arima_predictions

# Extend the years array to include future years
future_years = np.array(range(max(subset['Year']) + 1, max(subset['Year']) + years_to_predict + 1)).reshape(-1, 1)

# Calculate predictions for future years
future_predictions = linear_model.predict(future_years)

# Merge current and future predictions
all_years = np.concatenate((subset[['Year']].values, future_years))
all_predictions = np.concatenate((predictions, future_predictions))

import plotly.graph_objects as go

# Create a DataFrame for the chart
chart_data = pd.DataFrame({
    'Actual': np.concatenate((subset['new&used'].values, [None]*len(future_years))),
    'Predicted': all_predictions
}, index=np.concatenate((subset[['Year']].values.flatten(), future_years.flatten())))

# Convert the index to integers
chart_data.index = chart_data.index.astype(int)

# Create an empty figure object
fig = go.Figure()

# Create the chart for the actual data
fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Actual'], mode='lines', name='Actual',line=dict(color="Blue")))

# Plot the predictions
fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Predicted'], mode='lines', name='Predicted', line=dict(color='red', dash='dash')))

# Set the axis titles and the chart title
fig.update_layout(
    title='Line Plot of cars',
    xaxis_title='Year',
    xaxis = dict(
        tickmode = 'array',
        tickvals = chart_data.index,
        ticktext = chart_data.index.astype(int)
    ),
    yaxis_title='Count of new&used',
    legend_title='Legend'
)

# Show the chart
st.plotly_chart(fig, use_container_width=True)
st.info("We chose to get only 2 model as a Linear Regression Model and an ARIMA model.")

# Write this instruction in your terminal:
# streamlit run "c:..."         #Link to our python file
