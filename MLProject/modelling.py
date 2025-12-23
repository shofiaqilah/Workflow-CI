# modelling.py
import pandas as pd
import mlflow
import mlflow.sklearn
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# MLflow autolog 
mlflow.set_experiment("Hotel Reservations")
mlflow.sklearn.autolog()

# Path Dataset 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR, 
    "Preprocessing", 
    "hotel_reservations_preprocessed.csv"
)
print("File exists:", os.path.exists(DATA_PATH))

#  Load data 
df = pd.read_csv(DATA_PATH)

# Pisahkan fitur dan target
X = df.drop(columns=["booking_status"])
y = df["booking_status"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = RandomForestClassifier(
    random_state=42,
    n_estimators=100
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Metrik evalusi
acc = accuracy_score(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

mlflow.log_metric("accuracy", acc)
mlflow.log_metric("precision", ps)
mlflow.log_metric("recall", rec)
mlflow.log_metric("f1_score", f1)

mlflow.sklearn.log_model(model, "model")

print("Training selesai")
print(f"Accuracy: {acc:.4f}")
    
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

joblib.dump(model, os.path.join(ARTIFACT_DIR, "model.pkl"))