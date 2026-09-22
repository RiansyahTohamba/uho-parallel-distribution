import sqlite3
import os

DB_FILE = 'database.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS maba (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nama TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            no_pendaftaran TEXT NOT NULL UNIQUE,
            pembayaran_status TEXT NOT NULL
        )
    ''')

    cursor.execute('SELECT COUNT(*) FROM maba')
    if cursor.fetchone()[0] == 0:
        sample_data = [
            ('Budi Santoso', 'budi@example.com', 'MABA001', 'eligible'),
            ('Siti Nurhaliza', 'siti@example.com', 'MABA002', 'eligible'),
            ('Ahmad Wijaya', 'ahmad@example.com', 'MABA003', 'not_eligible'),
            ('Rina Suryani', 'rina@example.com', 'MABA004', 'eligible'),
            ('Doni Prabowo', 'doni@example.com', 'MABA005', 'not_eligible'),
        ]

        cursor.executemany(
            'INSERT INTO maba (nama, email, no_pendaftaran, pembayaran_status) VALUES (?, ?, ?, ?)',
            sample_data
        )

    conn.commit()
    conn.close()

def get_eligible_maba():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM maba WHERE pembayaran_status = ?', ('eligible',))
    rows = cursor.fetchall()

    conn.close()

    return [dict(row) for row in rows]
