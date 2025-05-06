import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_excel(filepath)

def handle_missing_values(data):
    return data.fillna(data.mean())

def normalize_features(data, feature_columns):
    scaler = StandardScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    return data

def preprocess_data(df, feature_columns):
    X = df[feature_columns]
    y = df['Life']
    return X, y