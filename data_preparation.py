# data_preparation.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset(file_path):
    # Assuming you have a CSV file with features and labels
    data = pd.read_csv(file_path)

    # Assuming the last column is the label column
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    return X, y

def preprocess_data(X_train, X_test):
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def split_data(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
