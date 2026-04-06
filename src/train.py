import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

from src.data_preprocessing import load_data, preprocess, split_data

# Load data
df = load_data("data/raw/churn.csv")

# Preprocess
df = preprocess(df)

# Split
X_train, X_test, y_train, y_test = split_data(df)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")

# Save model
joblib.dump(model, "model.pkl")
