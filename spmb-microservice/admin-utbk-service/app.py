from flask import Flask, render_template, jsonify, request
import requests
from database import init_db, save_akun_utbk, get_all_akun_utbk, akun_exists
import string
import random
import os

app = Flask(__name__)

MABA_SERVICE_URL = os.getenv('MABA_SERVICE_URL', 'http://localhost:5001')

def generate_username(nama):
    cleaned_nama = ''.join(c.lower() if c.isalnum() else '' for c in nama)
    random_suffix = ''.join(random.choices(string.digits, k=4))
    return f"{cleaned_nama}{random_suffix}"

def generate_password():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8))

@app.route('/')
def index():
    accounts = get_all_akun_utbk()
    return render_template('index.html', accounts=accounts)

@app.route('/api/accounts', methods=['GET'])
def get_accounts():
    accounts = get_all_akun_utbk()
    return jsonify({
        'status': 'success',
        'data': accounts,
        'count': len(accounts)
    }), 200

@app.route('/generate-accounts', methods=['POST'])
def generate_accounts():
    try:
        response = requests.get(f'{MABA_SERVICE_URL}/camaba-eligible')

        if response.status_code != 200:
            return jsonify({
                'status': 'error',
                'message': 'Gagal mengakses service Pendaftaran Maba'
            }), 500

        data = response.json()
        maba_list = data.get('data', [])

        if not maba_list:
            return jsonify({
                'status': 'error',
                'message': 'Tidak ada maba yang eligible'
            }), 400

        generated_accounts = []
        skipped = []

        for maba in maba_list:
            if akun_exists(maba['email']):
                skipped.append({
                    'nama': maba['nama'],
                    'reason': 'Akun sudah ada'
                })
                continue

            username = generate_username(maba['nama'])
            password = generate_password()

            if save_akun_utbk(
                maba['id'],
                maba['nama'],
                maba['email'],
                maba['no_pendaftaran'],
                username,
                password
            ):
                generated_accounts.append({
                    'nama': maba['nama'],
                    'email': maba['email'],
                    'username': username,
                    'password': password
                })
            else:
                skipped.append({
                    'nama': maba['nama'],
                    'reason': 'Gagal menyimpan akun'
                })

        return jsonify({
            'status': 'success',
            'message': f'Generated {len(generated_accounts)} akun UTBK',
            'generated': generated_accounts,
            'skipped': skipped,
            'total_processed': len(maba_list)
        }), 200

    except requests.exceptions.ConnectionError:
        return jsonify({
            'status': 'error',
            'message': 'Tidak dapat terhubung ke service Pendaftaran Maba (pastikan berjalan di port 5001)'
        }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    init_db()
    print("Admin UTBK Service running on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
