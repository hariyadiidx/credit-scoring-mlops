import os
import time
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from prometheus_client import start_http_server, Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# 1. Konfigurasi Path Model
# Asumsi: script ada di folder 'Monitoring_dan_Logging', 
# maka harus naik 1 tingkat (..) untuk masuk ke folder 'Model'
model_path = os.path.join('..', 'Model', 'model.pkl')
model = joblib.load(model_path)

# 2. Definisikan Metrik Monitoring (Sesuai Kriteria 4)
REQUEST_COUNT = Counter('request_total', 'Total HTTP requests ke model')
PREDICTION_COUNT = Counter('prediction_total', 'Total prediksi yang dihasilkan')
ERROR_COUNT = Counter('error_total', 'Total error saat inferensi')
CPU_USAGE = Gauge('cpu_usage_percent', 'Simulasi penggunaan CPU (%)')
RESPONSE_TIME = Histogram('response_time_seconds', 'Waktu respons inferensi (detik)')

app = FastAPI(title="Credit Scoring API")

@app.on_event("startup")
def startup_event():
    # Menjalankan server metrik terpisah untuk Prometheus di port 8000
    start_http_server(8000)

@app.get("/")
def read_root():
    return {"message": "Sistem Serving Credit Scoring Aktif!"}

@app.post("/predict")
async def predict(request: Request):
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    try:
        # Mengambil data JSON dari request
        data = await request.json()
        df = pd.DataFrame([data])
        
        # Prediksi menggunakan model asli
        prediction = model.predict(df)
        
        # Update metrik
        PREDICTION_COUNT.inc()
        RESPONSE_TIME.observe(time.time() - start_time)
        CPU_USAGE.set(15.5) # Simulasi statis
        
        return {"status": "success", "prediction": int(prediction[0])}
    
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
