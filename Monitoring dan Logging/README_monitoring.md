# Monitoring dan Logging — Predictive Maintenance API

## Nama: Mohamad Dwi Rezky Desyafa

Folder ini berisi file-file yang diperlukan untuk **serving model**, **monitoring** menggunakan Prometheus, dan **visualisasi dashboard** menggunakan Grafana.

---

## Daftar File

| File/Folder | Deskripsi |
|-------------|-----------|
| `7.Inference.py` | Flask API untuk serving model prediksi |
| `2.prometheus.yml` | Konfigurasi Prometheus untuk scraping metrics |
| `3.prometheus_exporter.py` | Prometheus exporter (referensi & opsional) |
| `1.bukti_serving/` | Folder untuk screenshot bukti serving |
| `4.bukti monitoring Prometheus/` | Folder untuk screenshot monitoring Prometheus |
| `5.bukti monitoring Grafana/` | Folder untuk screenshot monitoring Grafana |
| `6.bukti alerting Grafana/` | Folder untuk screenshot alerting Grafana |

---

## 1. Cara Menjalankan Inference API

### Prasyarat

```bash
pip install flask prometheus-client joblib scikit-learn pandas numpy
```

### Menyiapkan Model

Pastikan file berikut tersedia di folder `Monitoring dan Logging/`:

1. `best_model.pkl` — Model terbaik hasil training (dari `Membangun_model/artifacts/`)
2. `preprocessor.pkl` — ColumnTransformer yang sudah di-fit (dari `Membangun_model/artifacts/`)
3. `feature_columns.json` — Daftar nama kolom fitur (dari `Membangun_model/artifacts/`)

Copy file dari folder `Membangun_model/artifacts/`:

```bash
cp ../Membangun_model/artifacts/best_model.pkl .
cp ../Membangun_model/artifacts/preprocessor.pkl .
cp ../Membangun_model/artifacts/feature_columns.json .
```

### Menjalankan API

```bash
cd "Monitoring dan Logging"
python 7.Inference.py
```

API akan berjalan di `http://localhost:5001`.

### Menguji Endpoint

**Health check:**
```bash
curl http://localhost:5001/
```

**Prediksi (contoh mesin normal):**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"Type":"L","Air temperature [K]":298.1,"Process temperature [K]":308.6,"Rotational speed [rpm]":1551,"Torque [Nm]":42.8,"Tool wear [min]":0}'
```

**Melihat metrics:**
```bash
curl http://localhost:5001/metrics
```

### Ambil Screenshot Bukti Serving

Setelah API berjalan, ambil screenshot yang menunjukkan:
- API berjalan di terminal
- Response dari endpoint `/predict`

Simpan di folder `1.bukti_serving/`.

---

## 2. Cara Menjalankan Prometheus

### Download Prometheus

Download dari: https://prometheus.io/download/

### Konfigurasi

Gunakan file `2.prometheus.yml` yang sudah disediakan:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "predictive_maintenance_api"
    static_configs:
      - targets: ["localhost:5001"]
```

### Menjalankan Prometheus

```bash
prometheus --config.file=2.prometheus.yml
```

Prometheus akan berjalan di `http://localhost:9090`.

### Query Prometheus untuk Setiap Metrik

| No | Metrik | Query Prometheus |
|----|--------|-----------------|
| 1 | Total Requests | `total_prediction_requests_total` |
| 2 | Successful Requests | `successful_prediction_requests_total` |
| 3 | Failed Requests | `failed_prediction_requests_total` |
| 4 | Latency | `prediction_latency_seconds_sum / prediction_latency_seconds_count` |
| 5 | Machine Failure | `predicted_machine_failure_total` |
| 6 | Normal Machine | `predicted_normal_machine_total` |

### Ambil Screenshot Monitoring Prometheus

Buka Prometheus UI (`http://localhost:9090`), jalankan setiap query, dan ambil screenshot:

| Screenshot | File |
|------------|------|
| Total requests | `4.bukti monitoring Prometheus/1.monitoring_total_requests.jpg` |
| Success requests | `4.bukti monitoring Prometheus/2.monitoring_success_requests.jpg` |
| Failed requests | `4.bukti monitoring Prometheus/3.monitoring_failed_requests.jpg` |
| Latency | `4.bukti monitoring Prometheus/4.monitoring_latency.jpg` |
| Machine failure | `4.bukti monitoring Prometheus/5.monitoring_machine_failure.jpg` |

---

## 3. Cara Setup Grafana

### Download Grafana

Download dari: https://grafana.com/grafana/download

### Menjalankan Grafana

```bash
grafana-server
```

Buka `http://localhost:3000` (default: admin/admin).

### Menghubungkan Grafana ke Prometheus

1. Buka **Configuration** → **Data Sources**
2. Klik **Add data source**
3. Pilih **Prometheus**
4. URL: `http://localhost:9090`
5. Klik **Save & Test**

---

## 4. Membuat Dashboard Grafana

### Nama Dashboard (WAJIB):

```
Monitoring Predictive Maintenance - Rezky Desyafa
```

### Panel yang Dibuat

Buat dashboard baru dengan minimal **5 panel**:

| No | Panel Title | Visualization | Query |
|----|-------------|---------------|-------|
| 1 | Total Prediction Requests | Stat / Time series | `total_prediction_requests_total` |
| 2 | Successful Prediction Requests | Stat / Time series | `successful_prediction_requests_total` |
| 3 | Failed Prediction Requests | Stat / Time series | `failed_prediction_requests_total` |
| 4 | Prediction Latency (avg) | Gauge / Time series | `rate(prediction_latency_seconds_sum[1m]) / rate(prediction_latency_seconds_count[1m])` |
| 5 | Machine Failure Predictions | Stat / Time series | `predicted_machine_failure_total` |
| 6 | Normal Machine Predictions | Stat / Time series | `predicted_normal_machine_total` |

### Langkah Membuat Dashboard

1. Klik **+** → **Dashboard**
2. Klik **Add Panel**
3. Masukkan query Prometheus
4. Pilih visualization type
5. Isi Panel Title
6. Klik **Apply**
7. Ulangi untuk setiap panel
8. Klik **Save Dashboard** → Nama: `Monitoring Predictive Maintenance - Rezky Desyafa`

### Ambil Screenshot Monitoring Grafana

| Screenshot | File |
|------------|------|
| Total requests | `5.bukti monitoring Grafana/1.monitoring_total_requests.jpg` |
| Success requests | `5.bukti monitoring Grafana/2.monitoring_success_requests.jpg` |
| Failed requests | `5.bukti monitoring Grafana/3.monitoring_failed_requests.jpg` |
| Latency | `5.bukti monitoring Grafana/4.monitoring_latency.jpg` |
| Machine failure | `5.bukti monitoring Grafana/5.monitoring_machine_failure.jpg` |

---

## 5. Membuat Alert Grafana

### Alert: High Machine Failure Prediction Alert

1. Buka **Alerting** → **Alert Rules**
2. Klik **Create alert rule**
3. Konfigurasi:
   - **Name:** `High Machine Failure Prediction Alert`
   - **Query:** `predicted_machine_failure_total`
   - **Condition:** `WHEN last() IS ABOVE 0`
   - **Evaluate every:** `1m`
   - **For:** `0m` (langsung fire)
4. Klik **Save**

### Alternatif Alert

- Alert jika `failed_prediction_requests_total > 0` (ada request gagal)
- Alert jika latency terlalu tinggi: `prediction_latency_seconds_sum / prediction_latency_seconds_count > 1`

### Ambil Screenshot Alerting

| Screenshot | File |
|------------|------|
| Rules alert | `6.bukti alerting Grafana/1.rules_high_machine_failure.jpg` |
| Notifikasi alert | `6.bukti alerting Grafana/2.notifikasi_high_machine_failure.jpg` |

---

## 6. Mengirim Request untuk Men-trigger Metrik

Untuk men-generate data metrik, kirim beberapa request prediksi:

```bash
# Request mesin normal
curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d '{"Type":"L","Air temperature [K]":298.1,"Process temperature [K]":308.6,"Rotational speed [rpm]":1551,"Torque [Nm]":42.8,"Tool wear [min]":0}'

# Request yang mungkin menghasilkan Machine Failure
curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d '{"Type":"H","Air temperature [K]":302.0,"Process temperature [K]":312.0,"Rotational speed [rpm]":2800,"Torque [Nm]":65.0,"Tool wear [min]":200}'

# Request dengan input salah (untuk trigger failed)
curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d '{"invalid":"data"}'
```

Kirim beberapa kali agar metrik terakumulasi dan muncul di Grafana.

---

## Daftar Screenshot yang Harus Dikumpulkan

| No | Screenshot | Lokasi |
|----|-----------|--------|
| 1 | Bukti serving API | `1.bukti_serving/` |
| 2 | Monitoring total requests (Prometheus) | `4.bukti monitoring Prometheus/1.monitoring_total_requests.jpg` |
| 3 | Monitoring success requests (Prometheus) | `4.bukti monitoring Prometheus/2.monitoring_success_requests.jpg` |
| 4 | Monitoring failed requests (Prometheus) | `4.bukti monitoring Prometheus/3.monitoring_failed_requests.jpg` |
| 5 | Monitoring latency (Prometheus) | `4.bukti monitoring Prometheus/4.monitoring_latency.jpg` |
| 6 | Monitoring machine failure (Prometheus) | `4.bukti monitoring Prometheus/5.monitoring_machine_failure.jpg` |
| 7 | Monitoring total requests (Grafana) | `5.bukti monitoring Grafana/1.monitoring_total_requests.jpg` |
| 8 | Monitoring success requests (Grafana) | `5.bukti monitoring Grafana/2.monitoring_success_requests.jpg` |
| 9 | Monitoring failed requests (Grafana) | `5.bukti monitoring Grafana/3.monitoring_failed_requests.jpg` |
| 10 | Monitoring latency (Grafana) | `5.bukti monitoring Grafana/4.monitoring_latency.jpg` |
| 11 | Monitoring machine failure (Grafana) | `5.bukti monitoring Grafana/5.monitoring_machine_failure.jpg` |
| 12 | Rules alert (Grafana) | `6.bukti alerting Grafana/1.rules_high_machine_failure.jpg` |
| 13 | Notifikasi alert (Grafana) | `6.bukti alerting Grafana/2.notifikasi_high_machine_failure.jpg` |
