# SPMB Microservices

Sistem microservices sederhana untuk Pendaftaran Maba dan Admin UTBK.

## Services

### 1. Pendaftaran Maba Service
Service untuk mengelola data pendaftaran mahasiswa baru dengan informasi pembayaran.

**Port**: 5001

**Endpoint**:
- `GET /camaba-eligible` - Dapatkan list maba yang sudah membayar (eligible)

### 2. Admin UTBK Service
Service admin untuk manage akun UTBK dengan interface HTML.

**Port**: 5000

**Endpoint**:
- `GET /` - Interface admin
- `POST /generate-accounts` - Generate akun UTBK dari data yang eligible

## Setup & Menjalankan

### Pendaftaran Maba Service
```bash
cd pendaftaran-maba-service
pip install -r requirements.txt
python app.py
```

### Admin UTBK Service
```bash
cd admin-utbk-service
pip install -r requirements.txt
python app.py
```

## Database
Kedua service menggunakan SQLite untuk menyimpan data.
