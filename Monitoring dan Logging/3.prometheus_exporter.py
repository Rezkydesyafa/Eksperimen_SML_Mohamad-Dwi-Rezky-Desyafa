"""
Prometheus Exporter for Predictive Maintenance API
Nama: Mohamad Dwi Rezky Desyafa

File ini berfungsi sebagai dokumentasi dan referensi exporter Prometheus.
Metrik Prometheus sudah terintegrasi langsung di dalam 7.Inference.py
melalui library prometheus-client.

Metrik yang di-expose melalui endpoint /metrics:
1. total_prediction_requests      - Counter: jumlah total request prediksi
2. successful_prediction_requests - Counter: jumlah request berhasil
3. failed_prediction_requests     - Counter: jumlah request gagal
4. prediction_latency_seconds     - Histogram: waktu pemrosesan prediksi
5. predicted_machine_failure_total - Counter: jumlah prediksi Machine Failure
6. predicted_normal_machine_total  - Counter: jumlah prediksi Normal

Cara menjalankan:
    1. Jalankan API: python 7.Inference.py
    2. Akses metrik: http://localhost:5001/metrics
    3. Prometheus akan men-scrape endpoint ini setiap 5 detik

Jika ingin menjalankan exporter secara terpisah (opsional),
gunakan script ini untuk menambahkan metrik custom.
"""

from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
import requests
import sys


# Metrik tambahan (opsional — bisa digunakan untuk monitoring tambahan)
api_health_status = Gauge(
    'api_health_status',
    'Health status of the Predictive Maintenance API (1=up, 0=down)'
)


def check_api_health(api_url="http://localhost:5001/"):
    """Cek status kesehatan API."""
    try:
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            api_health_status.set(1)
            return True
        else:
            api_health_status.set(0)
            return False
    except requests.exceptions.RequestException:
        api_health_status.set(0)
        return False


def main():
    """
    Jalankan exporter sebagai service tambahan (opsional).
    Metrik utama sudah ada di 7.Inference.py (/metrics endpoint).
    Exporter ini hanya untuk monitoring tambahan seperti health check.
    """
    exporter_port = 8000
    print(f"Starting Prometheus exporter on port {exporter_port}...")
    print(f"Metrics available at: http://localhost:{exporter_port}/metrics")
    
    start_http_server(exporter_port)
    
    print("Exporter started. Checking API health every 10 seconds...")
    
    while True:
        is_healthy = check_api_health()
        status = "UP" if is_healthy else "DOWN"
        print(f"API Health: {status}")
        time.sleep(10)


if __name__ == "__main__":
    main()
