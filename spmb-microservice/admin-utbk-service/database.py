import sqlite3
import os

DB_FILE = 'database.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS akun_utbk (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_maba INTEGER NOT NULL,
            nama TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            no_pendaftaran TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

def save_akun_utbk(id_maba, nama, email, no_pendaftaran, username, password):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        cursor.execute('''
            INSERT INTO akun_utbk (id_maba, nama, email, no_pendaftaran, username, password)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (id_maba, nama, email, no_pendaftaran, username, password))

        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_all_akun_utbk():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM akun_utbk ORDER BY created_at DESC')
    rows = cursor.fetchall()

    conn.close()

    return [dict(row) for row in rows]

def akun_exists(email):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM akun_utbk WHERE email = ?', (email,))
    exists = cursor.fetchone()[0] > 0

    conn.close()

    return exists
