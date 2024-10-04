import sqlite3
import os
from src.config import DB_PATH


def create_db():
    directory = os.path.dirname(DB_PATH)
    if not os.path.exists(directory):
        os.makedirs(directory)

    conn = sqlite3.connect(os.path.join(directory, "hack_embeddings.db"))
    c = conn.cursor()
    c.execute('''
            CREATE TABLE IF NOT EXISTS embeddings(
               uuid INTEGER PRIMARY KEY AUTOINCREMENT,
               video_id TEXT,
               embedding_video BLOB,
               embedding_audio BLOB
            )
    ''')

    conn.commit()
    conn.close()


def add_embeddings(video_id, embedding_video, embedding_audio):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO embeddings (video_id, embedding_video, embedding_audio) VALUES (?, ?, ?)',
              (video_id, embedding_video, embedding_audio))
    conn.commit()
    conn.close()


def get_all_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'SELECT uuid, video_id, embedding_video, embedding_audio FROM embeddings')
    data = c.fetchall()
    conn.close()
    return data

def get_row_by_video_id(video_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'SELECT uuid, video_id, embedding_video, embedding_audio FROM embeddings WHERE video_id = ?', (video_id,))
    data = c.fetchall()
    conn.close()
    return data

