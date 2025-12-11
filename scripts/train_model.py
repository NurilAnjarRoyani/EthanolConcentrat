# scripts/train_model.py

import os
import numpy as np
from tslearn.datasets import UCR_UEA_datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# ------------------------------
# 1. Setup folder
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------
# 2. Load dataset EthanolConcentration
# ------------------------------
dataset = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = dataset.load_dataset("EthanolConcentration")

print("Shape X_train:", X_train.shape)
print("Shape X_test :", X_test.shape)

# ------------------------------
# 3. Flatten time-series menjadi 1D per sampel
# Bentuk 3D -> 2D (samples, timesteps*channels)
# ------------------------------
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)
print("Shape setelah flatten:", X_train_flat.shape, X_test_flat.shape)

# ------------------------------
# 4. Normalisasi fitur
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled  = scaler.transform(X_test_flat)

# ------------------------------
# 5. Training Logistic Regression
# ------------------------------
logreg = LogisticRegression(
    solver="liblinear",
    max_iter=3000,
    multi_class="auto"
)
logreg.fit(X_train_scaled, y_train)

# ------------------------------
# 6. Simpan model & scaler ke folder models/
# ------------------------------
model_path = os.path.join(MODEL_DIR, "logreg_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

joblib.dump(logreg, model_path)
joblib.dump(scaler, scaler_path)

print("Model Logistic Regression berhasil disimpan ->", model_path)
print("Scaler berhasil disimpan ->", scaler_path)
