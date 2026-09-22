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

### Option 1: Menggunakan Docker Compose (Recommended)

```bash
# Dari root project directory
docker-compose -f spmb-microservice/docker-compose.yml up --build
```

Services akan berjalan:
- Pendaftaran Maba Service: http://localhost:5001
- Admin UTBK Service: http://localhost:5000

Untuk menghentikan services:
```bash
docker-compose -f spmb-microservice/docker-compose.yml down
```

### Option 2: Menjalankan Secara Manual

#### Pendaftaran Maba Service
```bash
cd spmb-microservice/pendaftaran-maba-service
pip install -r requirements.txt
python app.py
```

#### Admin UTBK Service
```bash
cd spmb-microservice/admin-utbk-service
pip install -r requirements.txt
python app.py
```

## Database
Kedua service menggunakan SQLite untuk menyimpan data.

## Akses Interface Admin
Setelah services berjalan, akses interface admin di: **http://localhost:5000**

## Architecture
- **Pendaftaran Maba Service**: REST API untuk data maba
- **Admin UTBK Service**: Web interface yang mengkonsumsi API Pendaftaran Maba Service
- **Komunikasi**: REST API (HTTP)
- **Database**: SQLite (file-based)
