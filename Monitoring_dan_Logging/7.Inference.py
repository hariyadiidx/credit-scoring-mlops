import os
import time
import joblib
import pandas as pd
import psutil
from fastapi import FastAPI, Request
from prometheus_client import start_http_server, Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# 1. Konfigurasi Path Model
model_path = 'Model/model.pkl'
model = joblib.load(model_path)

# 2. Definisikan 11 Metrik Monitoring (Termasuk Metrik Bisnis)
REQUEST_COUNT = Counter('request_total', 'Total HTTP requests')
PREDICTION_COUNT = Counter('prediction_total', 'Total prediksi berhasil')
ERROR_COUNT = Counter('error_total', 'Total error sistem')
CPU_USAGE = Gauge('cpu_usage_percent', 'Persentase penggunaan CPU')
RAM_USAGE = Gauge('ram_usage_percent', 'Persentase penggunaan RAM')
RESPONSE_TIME = Histogram('response_time_seconds', 'Waktu respon API (detik)')

# --- METRIK BISNIS BARU ---
CREDIT_APPROVED = Counter('credit_approved_total', 'Total kredit yang disetujui (Low Risk)')
CREDIT_REJECTED = Counter('credit_rejected_total', 'Total kredit yang ditolak (High Risk)')

MODEL_VERSION = Gauge('model_version', 'Versi model yang dipakai')
PREDICTION_LATENCY = Gauge('prediction_latency', 'Latensi inferensi model')
UPTIME_SECONDS = Gauge('uptime_seconds', 'Waktu aplikasi aktif (detik)')

app = FastAPI(title="Credit Scoring API")
start_time_app = time.time()

@app.on_event("startup")
def startup_event():
    # Menjalankan server metrik terpisah untuk Prometheus di port 8000
    start_http_server(8000)

@app.get("/")
def read_root():
    # Halaman utama agar tidak muncul pesan "Not Found"
    return {"message": "Sistem API Credit Scoring Aktif dan Berjalan!"}

@app.post("/predict")
async def predict(request: Request):
    start_predict = time.time()
    REQUEST_COUNT.inc()
    
    try:
        data = await request.json()
        df = pd.DataFrame([data])
        
        # Proses Prediksi oleh Model Machine Learning
        prediction = model.predict(df)
        hasil_prediksi = int(prediction[0])
        
        # --- LOGIKA METRIK BISNIS ---
        # Asumsi: 1 = Disetujui (Low Risk), 0 = Ditolak (High Risk)
        if hasil_prediksi == 1:
            CREDIT_APPROVED.inc()
        else:
            CREDIT_REJECTED.inc()
        # -----------------------------
        
        # Update Metrik Infrastruktur & Performa
        PREDICTION_COUNT.inc()
        RESPONSE_TIME.observe(time.time() - start_predict)
        
        # Update data hardware asli sistem
        CPU_USAGE.set(psutil.cpu_percent()) 
        RAM_USAGE.set(psutil.virtual_memory().percent)
        
        MODEL_VERSION.set(1.0)
        PREDICTION_LATENCY.set(time.time() - start_predict)
        UPTIME_SECONDS.set(time.time() - start_time_app)
        
        return {"status": "success", "prediction": hasil_prediksi}
    
    except Exception as e:
        ERROR_COUNT.inc()
        return {"status": "error", "message": str(e)}

@app.get("/metrics")
def metrics():
    # Endpoint untuk Prometheus mengambil data monitoring
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == '__main__':
    import uvicorn
    print("Menjalankan Model Serving FastAPI di port 5002...")
    uvicorn.run(app, host="127.0.0.1", port=5002)