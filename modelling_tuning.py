import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import json

# ==========================================
# 1. KONFIGURASI DAGSHUB & MLFLOW
# ==========================================
DAGSHUB_USERNAME = "hariyadiidx" 
DAGSHUB_REPO = "Credit-Scoring-MLflow"

dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")

# ==========================================
# 2. PERSIAPAN DATA
# ==========================================
print("Memuat dataset sementara...")
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy')

# ==========================================
# 3. EKSPERIMEN MLFLOW MANUAL
# ==========================================
mlflow.set_experiment("Credit_Scoring_Tuning_Experiment")

with mlflow.start_run():
    print("Memulai proses Hyperparameter Tuning dengan Grid Search...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)

    print("Membuat struktur folder model dan artefak...")
    
    # --- TRIK RAHASIA (PASTI BERHASIL) ---
    # 1. Buat folder 'model' fisiknya di lokal Codespaces dulu
    if os.path.exists("model_lokal"):
        shutil.rmtree("model_lokal") # Hapus jika sudah ada sisa sebelumnya
    mlflow.sklearn.save_model(sk_model=best_model, path="model_lokal")
    
    # 2. Dorong paksa seluruh folder lokal itu ke DagsHub!
    mlflow.log_artifacts(local_dir="model_lokal", artifact_path="model")
    
    # --- Artefak 1 (Gambar) ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix Hasil Tuning')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = "training_confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path) 
    
    # --- Artefak 2 (JSON) ---
    metric_info = {"model_accuracy": accuracy, "best_params": grid_search.best_params_}
    json_path = "metric_info.json"
    with open(json_path, "w") as f:
        json.dump(metric_info, f)
    mlflow.log_artifact(json_path)

    print("Selesai! Cek eksperimen BARU di DagsHub Anda.")