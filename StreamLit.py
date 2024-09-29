import streamlit as st
from DB_code import add_embedding_video_test, get_video_embeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from duplicate_video_yappy.parser import download_file
import torch
from duplicate_video_yappy.models.model import SimilarityRecognizer
from duplicate_video_yappy.src.video_analysis import get_video_features
from duplicate_video_yappy.src.preprocess import Preprocess


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
    video = download_file(video_link)
    # Извлечение эмбеддинга(надо написать)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video_model = SimilarityRecognizer(model_type="base", batch_size=8).to(device)
    video_model.load_pretrained_weights("checkpoints/best_model_base_224_16x16_rgb.pth")
    video_model.eval()
    preprocess = Preprocess(clip_len=8, out_size=224, frame_interval=1, channels=1)
    embedding = get_video_features(preprocess, video_model, video_paths=video)

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
        st.markdown(f'<p class="result">Это дубликат видео под ID: {most_similar_id}</p>', unsafe_allow_html=True)

    else:
        st.markdown('<p class="result">Похожих видео не найдено.</p>', unsafe_allow_html=True)
