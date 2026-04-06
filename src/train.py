import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def train_model(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
