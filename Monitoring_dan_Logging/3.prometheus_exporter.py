
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# 1. COUNTERS
REQUEST_COUNT = Counter('request_total', 'Total HTTP requests HTTP ke model')
ERROR_COUNT = Counter('error_total', 'Total error saat inferensi')
PREDICTION_COUNT = Counter('prediction_total', 'Total prediksi yang dihasilkan')

# 2. GAUGES
CPU_USAGE = Gauge('cpu_usage_percent', 'Simulasi penggunaan CPU (%)')
MEMORY_USAGE = Gauge('memory_usage_mb', 'Simulasi penggunaan RAM (MB)')
DATA_DRIFT = Gauge('data_drift_score', 'Indikator Data Drift (Jarak sebaran data)')
MODEL_CONFIDENCE = Gauge('model_confidence_score', 'Skor probabilitas / keyakinan model')
FEATURE_NAN = Gauge('feature_nan_count', 'Jumlah fitur kosong (NaN) dari request')

# 3. HISTOGRAMS
RESPONSE_TIME = Histogram('response_time_seconds', 'Waktu respons inferensi (detik)')
BATCH_SIZE = Histogram('batch_size', 'Jumlah data per request prediksi')

def start_metrics_server(port=8000):
    start_http_server(port)
    print(f"✅ Prometheus metrics exporter berjalan di port {port}")

import time

if __name__ == '__main__':
    start_metrics_server(8000)
    while True:
        time.sleep(1)
