import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub



# Defining the path to my CSV file
file_path = r"C:\Users\thoma\OneDrive\Documents\Exeter Year 1\Data Science project\Final_Project\Bean Data Python.csv"


chocolate_df = pd.read_csv(file_path)


print(chocolate_df.head())

print(chocolate_df.columns)

import pandas as pd
import matplotlib.pyplot as plt

file_path = r"C:\Users\thoma\OneDrive\Documents\Exeter Year 1\Data Science project\Final_Project\Bean Data.csv"
chocolate_df = pd.read_csv(file_path)

# Here, I am combining "Latin America" and "South America" into one region named "South America"
chocolate_df['Region'] = chocolate_df['Region'].replace({
    'Latin America': 'South America',
    'south america': 'South America' 
})

# I am Grouping by 'Region' and calculating the average rating
region_avg = chocolate_df.groupby('Region')['Rating'].mean().reset_index()

region_avg = region_avg.sort_values(by='Rating', ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(region_avg['Region'], region_avg['Rating'], color='mediumseagreen')
plt.ylim(3, 3.50)  # Adjust the y-axis range
plt.xlabel('Region')
plt.ylabel('Average Rating')
plt.title('Average Chocolate Rating by Region')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = r"C:\Users\thoma\OneDrive\Documents\Exeter Year 1\Data Science project\Final_Project\Bean Data.csv"
chocolate_df = pd.read_csv(file_path)

plt.figure(figsize=(12, 6))

# Creating the histogram with seaborn
hist_plot = sns.histplot(chocolate_df['Rating'], bins=8, kde=False, color='chocolate', alpha=0.6, label='Histogram')

# Adding data labels on top of bars
for patch in hist_plot.patches:
    count = int(patch.get_height())  # Get bar height (frequency)
    if count > 0:  # Add label if count is non-zero
        plt.text(patch.get_x() + patch.get_width() / 2, count + 1,  # Slightly above the bar
                 str(count), ha='center', fontsize=10, color='black')

# Adding gridlines for clarity
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Setting labels and title
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Chocolate Ratings')

# Adjusting the legend position for better alignment
plt.legend(loc='upper right')  # Positioned logically away from the x-axis

# Ensuring the plot is visually appealing
plt.tight_layout()
plt.show()

 
#idea- webscrape data on prices- do graph price to rating? maybe scrape additional data on beans
#do randomforest model
#use gitbash and create git repoitory



#webscraping

import sqlite3
from bs4 import BeautifulSoup
import requests
import pandas as pd

page_to_scrape = requests.get('https://www.indexmundi.com/commodities/?commodity=cocoa-beans&months=240')
soup = BeautifulSoup(page_to_scrape.text, "html.parser")

# Extracting table headers
columns = [header.text.strip() for header in soup.find_all("th")]

# Extracting table rows
rows = []
for row in soup.find_all("tr"):
    cells = row.find_all("td")
    if len(cells) == len(columns):  # Ensure row matches header count
        rows.append([cell.text.strip() for cell in cells])

df = pd.DataFrame(rows, columns=columns)

# Here I am turning 'Month' column into datetime objects
df['Month'] = pd.to_datetime(df['Month'], format='%b %Y', errors='coerce')

df = df[df['Month'].between('2005-01-01', '2024-12-31')]

print(df.head())
print(df.shape)

# Saving the DataFrame to SQLite database
try:
    conn = sqlite3.connect('C:\\Users\\thoma\\OneDrive\\Documents\\Exeter Year 1\\Data Science project\\Final_Project\\beans.db')
    print(conn)  # Here, I am verifying the connection
    df.to_sql('Prices', conn, if_exists='append', index=False)
    conn.close()
except Exception as e:
    print(f"Error occurred: {e}")

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Example data preprocessing
# Ensure 'Price' column is cleaned and numeric
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna(subset=['Price'])

# Feature engineering: create lag features and rolling averages
df['Lag_1'] = df['Price'].shift(1)
df['Lag_3'] = df['Price'].shift(3)
df['Rolling_Avg_3'] = df['Price'].rolling(3).mean()
df = df.dropna()  # Drop rows with NaN values due to shifting

# Define features (X) and target (y)
X = df[['Lag_1', 'Lag_3', 'Rolling_Avg_3']]
y = df['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("R-squared Score:", r2)

# Predict future price (example)
future_features = np.array([[df['Price'].iloc[-1], df['Price'].iloc[-3], df['Price'].iloc[-3:].mean()]])
future_price = model.predict(future_features)
print("Predicted Future Price:", future_price)

import matplotlib.pyplot as plt
import pandas as pd

# Extract the indices of the test data
test_indices = X_test.index

# Retrieve the corresponding months from the original DataFrame
test_months = df.loc[test_indices, 'Month']

# Create a DataFrame of actual vs. predicted prices with corresponding months
results_df = pd.DataFrame({
    'Month': test_months,
    'Actual Price': y_test.values,
    'Predicted Price': y_pred
})

# Sort results by month for better visualization
results_df = results_df.sort_values(by='Month')

# Plot actual vs. predicted prices
plt.figure(figsize=(12, 8))
plt.plot(results_df['Month'], results_df['Actual Price'], label='Actual Prices', color='blue', marker='o')
plt.plot(results_df['Month'], results_df['Predicted Price'], label='Predicted Prices', color='orange', marker='x')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.xlabel("Month")
plt.ylabel("Price")
plt.title("Actual vs. Predicted Cocoa Bean Prices Over Time")
plt.legend()
plt.grid(True)

# Annotate points with Month
for i in range(len(results_df)):
    plt.text(results_df['Month'].iloc[i], results_df['Actual Price'].iloc[i], 
             results_df['Month'].iloc[i].strftime('%b %Y'), fontsize=8, alpha=0.7)

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd

# Example: Simulating features for September 2027
# Assume the most recent price (and features) available in your dataset
last_price = df['Price'].iloc[-1]  # The most recent price
lag_1 = last_price  # September 2027's lag_1 would be the last price in the dataset
lag_3 = df['Price'].iloc[-3]  # Lag_3 would be the price 3 steps before the last one
rolling_avg_3 = df['Price'].iloc[-3:].mean()  # Rolling average of the last 3 months

# Create feature array for the prediction
future_features = pd.DataFrame([[lag_1, lag_3, rolling_avg_3]], columns=['Lag_1', 'Lag_3', 'Rolling_Avg_3'])

# Predict the cocoa bean price for September 2027
predicted_price_2027 = model.predict(future_features)
print("Predicted Cocoa Bean Price for September 2027:", predicted_price_2027[0])
