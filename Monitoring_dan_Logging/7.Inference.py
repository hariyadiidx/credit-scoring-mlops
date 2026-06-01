# 7.Inference.py
from fastapi import FastAPI, Response
import uvicorn
import time
import random
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

# ==========================================
# DEFINISI 11 METRIK (Sesuai Screenshot Anda)
# ==========================================
REQUEST_TOTAL = Counter('request_total', 'Total HTTP requests')
PREDICTION_TOTAL = Counter('prediction_total', 'Total predictions made')
ERROR_TOTAL = Counter('error_total', 'Total prediction errors')

CPU_USAGE = Gauge('cpu_usage_percent', 'Simulated CPU usage percentage')
RAM_USAGE = Gauge('ram_usage_percent', 'Simulated RAM usage percentage')

# Menggunakan Counter agar namanya persis "response_time_seconds_count"
RESPONSE_TIME_COUNT = Counter('response_time_seconds_count', 'Count of response times')

CREDIT_APPROVED = Counter('credit_approved_total', 'Total credit approved')
CREDIT_REJECTED = Counter('credit_rejected_total', 'Total credit rejected')

MODEL_VERSION = Gauge('model_version', 'Current model version')
PREDICTION_LATENCY = Gauge('prediction_latency', 'Latency of the last prediction in seconds')
UPTIME_SECONDS = Gauge('uptime_seconds', 'Server uptime in seconds')

# Setup nilai awal
START_TIME = time.time()
MODEL_VERSION.set(1.0) # Set versi model ke 1 sesuai gambar Anda

@app.get("/predict")
async def predict():
    start_time = time.time()
    REQUEST_TOTAL.inc()
    RESPONSE_TIME_COUNT.inc()
    
    try:
        # Simulasi waktu komputasi model (bervariasi antara 0.005 - 0.05 detik)
        latency = random.uniform(0.005, 0.05)
        time.sleep(latency)
        
        # Simulasi hasil Credit Scoring (70% Approved, 30% Rejected)
        is_approved = random.random() < 0.70
        PREDICTION_TOTAL.inc()
        
        if is_approved:
            CREDIT_APPROVED.inc()
            hasil = "Approved"
        else:
            CREDIT_REJECTED.inc()
            hasil = "Rejected"
            
        # Simulasi fluktuasi CPU dan RAM untuk grafik Grafana
        CPU_USAGE.set(random.uniform(2.0, 15.0)) # Simulasi CPU 2% - 15%
        RAM_USAGE.set(random.uniform(25.0, 35.0)) # Simulasi RAM sekitar 28% sesuai gambar
        
        # Update Latency
        PREDICTION_LATENCY.set(latency)
        
        return {
            "status": "success", 
            "prediction": hasil,
            "latency_seconds": round(latency, 4)
        }

    except Exception as e:
        ERROR_TOTAL.inc()
        return {"status": "error", "message": str(e)}

@app.get("/metrics")
def metrics():
    # Update uptime setiap kali Prometheus mengambil data (scrape)
    UPTIME_SECONDS.set(time.time() - START_TIME)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == '__main__':
    print("Menjalankan API Credit Scoring dengan 11 Metrik di Port 5003...")
    uvicorn.run(app, host="0.0.0.0", port=5003)