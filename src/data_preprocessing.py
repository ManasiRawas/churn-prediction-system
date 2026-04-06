import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.dropna()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df = df.dropna()

    df = df.drop("customerID", axis=1)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    return df

def split_data(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(X, y, test_size=0.2, random_state=42)
