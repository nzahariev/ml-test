# generate_synthetic_data.py
import pandas as pd
import numpy as np

# Create a synthetic dataset with additional feature for sea view
np.random.seed(42)

# Generate random data
num_samples = 1000
bedrooms = np.random.randint(1, 5, size=num_samples)
square_footage = np.random.randint(800, 2500, size=num_samples)
distance_to_city_center = np.random.uniform(1, 20, size=num_samples)
sea_view = np.random.choice([0, 1], size=num_samples)
noise = np.random.normal(0, 20000, size=num_samples)

# Generate target variable (price) using a simple formula
price = 50000 + 300 * bedrooms + 100 * square_footage + 500 * distance_to_city_center + 20000 * sea_view + noise

# Create a DataFrame
data = pd.DataFrame({
    'bedrooms': bedrooms,
    'square_footage': square_footage,
    'distance_to_city_center': distance_to_city_center,
    'sea_view': sea_view,
    'price': price
})

# Save the dataset to a CSV file
data.to_csv('synthetic_real_estate_data.csv', index=False)

# Display the first few rows of the synthetic dataset
print(data.head())
