from fastapi import FastAPI
import uvicorn
import time
import random
import importlib.util
import sys

# Memuat module prometheus_exporter dari file lokal
sys.path.append('.')
exporter_module = importlib.import_module("3.prometheus_exporter")

app = FastAPI(title="Credit Scoring API")

@app.on_event("startup")
def startup_event():
    # Menjalankan metrics server di port 8000 (port yang akan di-scrape Prometheus)
    exporter_module.start_metrics_server(8000)

@app.get("/")
def read_root():
    return {"message": "Sistem Serving Credit Scoring Aktif!"}

@app.post("/predict")
def predict():
    start_time = time.time()
    exporter_module.REQUEST_COUNT.inc()

    try:
        # Simulasi proses inferensi model (Dummy prediction untuk memantik metriks)
        exporter_module.PREDICTION_COUNT.inc()
        exporter_module.BATCH_SIZE.observe(1)

        # Simulasi fluktuasi metrik sistem & model (agar grafik Grafana bergerak)
        exporter_module.CPU_USAGE.set(random.uniform(10.0, 75.0))
        exporter_module.MEMORY_USAGE.set(random.uniform(150.0, 450.0))
        exporter_module.DATA_DRIFT.set(random.uniform(0.01, 0.15))
        exporter_module.MODEL_CONFIDENCE.set(random.uniform(0.70, 0.99))
        exporter_module.FEATURE_NAN.set(random.randint(0, 1))

        # Mencatat waktu respons
        exporter_module.RESPONSE_TIME.observe(time.time() - start_time)

        return {"status": "success", "prediction": random.choice([0, 1])}

    except Exception as e:
        exporter_module.ERROR_COUNT.inc()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("Menjalankan Model Serving FastAPI di port 5002...")
    uvicorn.run(app, host="127.0.0.1", port=5002)
