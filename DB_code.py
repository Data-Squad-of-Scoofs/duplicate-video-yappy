import sqlite3
import streamlit as st


def create_db():
    conn = sqlite3.connect("hack_embeddings.db")
    c = conn.cursor()
    c.execute('''
            CREATE TABLE IF NOT EXIST embeddings(
               uuid TEXT PRIMARY KEY AUTOINCREMENT,
               video_link TEXT,
               embedding_video TEXT,
               embedding_audio TEXT
            )
    ''')

    conn.commit()
    conn.close()


def add_embedding_video_test(video_link, embedding):
    conn = sqlite3.connect('video_embeddings.db')
    c = conn.cursor()
    c.execute('INSERT INTO embeddings (video_link, embedding_video) VALUES (?, ?)', (video_link, embedding))
    conn.commit()
    conn.close()


def add_embedding_video_train(uuid, embedding):
    conn = sqlite3.connect('video_embeddings.db')
    c = conn.cursor()
    c.execute('INSERT INTO embeddings (uuid, embedding_video) VALUES (?, ?)', (uuid, embedding))
    conn.commit()
    conn.close()


def add_embedding_audio_test(video_link, embedding):
    conn = sqlite3.connect('video_embeddings.db')
    c = conn.cursor()
    c.execute('INSERT INTO embeddings (video_link, embedding_audio) VALUES (?, ?)', (video_link, embedding))
    conn.commit()
    conn.close()


def add_embedding_audio_train(uuid, embedding):
    conn = sqlite3.connect('video_embeddings.db')
    c = conn.cursor()
    c.execute('INSERT INTO embeddings (uuid, embedding_audio) VALUES (?, ?)', (uuid, embedding))
    conn.commit()
    conn.close()


@st.cache_data
def get_video_embeddings():
    conn = sqlite3.connect('video_embeddings.db')
    c = conn.cursor()
    c.execute('SELECT uuid, video_link, embedding_video FROM embeddings')
    data = c.fetchall()
    conn.close()
    return data


@st.cache_data
def get_audio_embeddings(most_similar_id):
    conn = sqlite3.connect('video_embeddings.db')
    c = conn.cursor()
    c.execute("SELECT audio_embeddings FROM your_table WHERE uuid=?", (most_similar_id,))
    result = c.fetchone()
    conn.close()
    return result
