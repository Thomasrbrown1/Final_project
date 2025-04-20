import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

import pandas as pd
import pandas as pd

# Define the path to your CSV file
file_path = r"C:\Users\thoma\OneDrive\Documents\Exeter Year 1\Data Science project\Final_Project\Bean Data Python.csv"

# Load the CSV file into a DataFrame
chocolate_df = pd.read_csv(file_path)

# Print the first few rows of the DataFrame
print(chocolate_df.head())

print(chocolate_df.columns)

import pandas as pd
import matplotlib.pyplot as plt

# Define the path to your dataset file
file_path = r"C:\Users\thoma\OneDrive\Documents\Exeter Year 1\Data Science project\Final_Project\Bean Data.csv"

# Load the CSV file into a DataFrame
chocolate_df = pd.read_csv(file_path)

# Group by 'Region' and calculate the average rating
region_avg = chocolate_df.groupby('Region')['Rating'].mean().reset_index()

# Sort the regions by average rating for better visualization
region_avg = region_avg.sort_values(by='Rating', ascending=False)

# Plot the bar chart with improved styling
plt.figure(figsize=(12, 6))
plt.bar(region_avg['Region'], region_avg['Rating'], color='mediumseagreen')
plt.ylim(3, 3.50)  # Adjust the y-axis range
plt.xlabel('Region')
plt.ylabel('Average Rating')
plt.title('Average Chocolate Rating by Region')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show the plot
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to your dataset file
file_path = r"C:\Users\thoma\OneDrive\Documents\Exeter Year 1\Data Science project\Final_Project\Bean Data.csv"

# Load the CSV file into a DataFrame
chocolate_df = pd.read_csv(file_path)

# Create the histogram and density plot
plt.figure(figsize=(12, 6))

# Histogram
hist_plot = sns.histplot(chocolate_df['Rating'], bins=10, kde=False, color='chocolate', alpha=0.6, label='Histogram')

# Add data labels on top of bars
for patch in hist_plot.patches:
    count = int(patch.get_height())  # Get bar height (frequency)
    if count > 0:  # Add label if count is non-zero
        plt.text(patch.get_x() + patch.get_width() / 2, count + 1,  # Slightly above the bar
                 str(count), ha='center', fontsize=10, color='black')

# Density Plot
sns.kdeplot(chocolate_df['Rating'], color='blue', lw=2, label='Density')

# Add gridlines for clarity
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Set plot labels and title
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Chocolate Ratings')
plt.legend()

# Tight layout for aesthetics
plt.tight_layout()

# Show the plot
plt.show()
 
#idea- webscrape data on prices- do graph price to rating? maybe scrape additional data on beans
#do randomforest model
#use gitbash and create git repoitory



#webscraping
# Web scraping and saving data to SQLite
import sqlite3
from bs4 import BeautifulSoup
import requests
import pandas as pd

# Fetch the webpage with extended data
page_to_scrape = requests.get('https://www.indexmundi.com/commodities/?commodity=cocoa-beans&months=240')
soup = BeautifulSoup(page_to_scrape.text, "html.parser")

# Extract table headers
columns = [header.text.strip() for header in soup.find_all("th")]

# Extract table rows
rows = []
for row in soup.find_all("tr"):
    cells = row.find_all("td")
    if len(cells) == len(columns):  # Ensure row matches header count
        rows.append([cell.text.strip() for cell in cells])

# Create DataFrame
df = pd.DataFrame(rows, columns=columns)

# Parse 'Month' column into datetime objects
df['Month'] = pd.to_datetime(df['Month'], format='%b %Y', errors='coerce')

# Filter rows for data between 2005 and 2024
df = df[df['Month'].between('2005-01-01', '2024-12-31')]

# Debugging the DataFrame
print(df.head())
print(df.shape)

# Save the DataFrame to SQLite database
try:
    conn = sqlite3.connect('C:\\Users\\thoma\\OneDrive\\Documents\\Exeter Year 1\\Data Science project\\Final_Project\\beans.db')
    print(conn)  # Verify connection
    df.to_sql('Prices', conn, if_exists='append', index=False)
    conn.close()
except Exception as e:
    print(f"Error occurred: {e}")
