import streamlit as st
from DB_code import add_embedding_video_test, get_video_embeddings, get_audio_embeddings
from utils import extract_audio_embedding, load_model, extract_video_embedding, create_audio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Установка стилей
st.set_page_config(page_title="Сервис распознавания видео", layout="wide")
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #4B0082;
    }
    .description {
        font-size: 18px;
        color: #555555;
    }
    .result {
        font-size: 20px;
        font-weight: bold;
        color: #008000;
    }
    </style>
""", unsafe_allow_html=True)

# Можно использовать @st.cache_data перед load data которая по идеи нужна для загрузки эмбедингов чтобы сравнить
st.title("Тест сервиса по распознаванию видео", anchor=None)
st.markdown('<p class="description">Введите ссылку на видео для проверки на дубликат.</p>', unsafe_allow_html=True)

video_link = st.text_input("Ссылка на видео:")

if st.button("Отправить"):
    # Тут должен быть парсинг видео

    # Извлечение эмбеддинга(надо написать)
    embedding = extract_video_embedding(video)

    # Получаем все эмбеддинги из базы данных
    embeddings_data = get_video_embeddings()

    # Добавление эмбеддинга в базу данных
    add_embedding_video_test(video_link, embedding.tobytes())

    max_similarity = -1
    most_similar_id = None
    threshold = 0.8601731272679782

    for id, link, stored_embedding in embeddings_data:
        stored_embedding = np.frombuffer(stored_embedding, dtype=np.float64).reshape(1, -1)
        similarity = cosine_similarity(embedding, stored_embedding)[0][0]

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_id = id
            break

    if max_similarity > threshold:
        st.markdown(f'<p class="result">Самое похожая видеодорожка у ID: {most_similar_id}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="description"Идёт проверка на схожесть аудиодорожки></p>', unsafe_allow_html=True)
        audio = create_audio(video)
        audio_embedding = get_audio_embeddings(most_similar_id)
        similar_audio_embedding = model.get_embeedding(audio)
        audio_similarity = cosine_similarity(audio_embedding, similar_audio_embedding)[0][0]
        if audio_similarity > 0:
            st.markdown('<p class="result">Это дубликат</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="result">Исходя из аудидорожки это не дубликат.</p>', unsafe_allow_html=True)

    else:
        st.markdown('<p class="result">Похожих видео не найдено.</p>', unsafe_allow_html=True)
