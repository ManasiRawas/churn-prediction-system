# Customer Churn Prediction System 🚀

End-to-end Machine Learning project to predict customer churn using real-world telecom data.

## 🔥 Features

* Data preprocessing pipeline
* Machine learning model training
* REST API using FastAPI
* Interactive dashboard using Streamlit

## 🛠 Tech Stack

* Python
* Pandas, Scikit-learn
* FastAPI
* Streamlit

## 📂 Project Structure

```
churn-prediction-system/
│
├── api/                # FastAPI app
├── dashboard/          # Streamlit UI
├── src/                # ML pipeline
├── data/               # Dataset
├── model.pkl           # Trained model
└── requirements.txt
```

## 🚀 How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run training

python src/train.py

### 3. Run API

uvicorn api.main:app --reload

### 4. Run dashboard

streamlit run dashboard/app.py

## 🎯 Output

* Predict whether a customer will churn or not

## 📌 Author

Manasi Rawas
