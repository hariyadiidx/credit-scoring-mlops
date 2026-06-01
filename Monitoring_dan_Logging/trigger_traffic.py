# trigger_traffic.py
import requests
import time
import random

API_URL = "http://127.0.0.1:5003/predict"

print("🚀 Memulai Simulasi Traffic Credit Scoring...")
print("Biarkan terminal ini menyala agar grafik Grafana Anda bergerak naik!")

count = 0
while True:
    try:
        response = requests.get(API_URL)
        data = response.json()
        count += 1
        print(f"[{count}] Hasil: {data['prediction']} | Waktu: {data['latency_seconds']}s")
        
        # Jeda sangat cepat agar angkanya cepat naik menjadi ribuan
        time.sleep(random.uniform(0.01, 0.1))
        
    except Exception as e:
        print("API belum menyala atau error...")
        time.sleep(2)