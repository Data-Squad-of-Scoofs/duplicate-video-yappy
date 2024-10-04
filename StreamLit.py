import streamlit as st
from src.utils import download_file
import torch
import os
import pickle

from DB_code import add_embeddings, get_all_data, get_row_by_uuid, create_db, get_audio_embedding_by_uuid
from src.video_analysis import get_video_features
from src.audio_analysis import get_audio_features, load_and_preprocess_audio
from src.video_preprocess import load_and_preprocess_video
from src.utils import get_video_model, get_audio_model, find_most_similar_by_video, compute_similarity
from src.config import VIDEO_SIMILARITY_THRESHOLD, AUDIO_SIMILARITY_THRESHOLD


device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
st.markdown('<p class="description">Введите ссылку на видео для проверки на дубликат.</p>',
            unsafe_allow_html=True)

video_link = st.text_input("Ссылка на видео:")

@st.cache_resource
def load_models():
    video_model = get_video_model(device)
    audio_model = get_audio_model(device)
    return video_model, audio_model

video_model, audio_model = load_models()

create_db()

if st.button("Отправить"):
    video_id = video_link.split('/')[-1].split('.')[0]

    video_path = f'temp_downloads/{video_id}.mp4'

    download_file(video_link, video_path)
    frames = load_and_preprocess_video(video_path)


    if frames is not None:
        query_video_embedding = get_video_features(frames, video_model)
    else:
        raise ValueError('Не удалось загрузить видео :/')

    data = get_all_data()

    db_embeddings = [
        (
            row[0], 
            pickle.loads(row[1]) if row[1] is not None else None, 
            pickle.loads(row[2]) if row[2] is not None else None
        )
        for row in data 
    ]

    similar_uuid, video_similarity_rate = find_most_similar_by_video(
        query_video_embedding, db_embeddings)

    if 0 < video_similarity_rate < VIDEO_SIMILARITY_THRESHOLD:
        st.markdown('<p class="result">Похожих видео не найдено.</p>',
                            unsafe_allow_html=True)

        video_embedding_serialized = pickle.dumps(query_video_embedding)

        audio_data = load_and_preprocess_audio(video_path)
        query_audio_embeddings = get_audio_features(audio_data, audio_model)
        if query_audio_embeddings is not None:
            audio_embedding_serialized = pickle.dumps(query_video_embedding)
            add_embeddings(video_embedding_serialized, audio_embedding_serialized)
        else:
            add_embeddings(video_embedding_serialized, None)


    else:    
        audio_data = load_and_preprocess_audio(video_path)
        query_audio_embeddings = get_audio_features(audio_data, audio_model)
        db_similar_video_data = get_row_by_uuid(similar_uuid)
        if db_similar_video_data:
            db_audio_embeddings = get_audio_embedding_by_uuid(similar_uuid)[0]

            db_audio_embeddings = pickle.loads(db_audio_embeddings) if db_audio_embeddings is not None else None

            if query_audio_embeddings is not None and db_audio_embeddings is not None:
                audio_similarity = compute_similarity(query_audio_embeddings, db_audio_embeddings)

                if audio_similarity < AUDIO_SIMILARITY_THRESHOLD:
                    video_embedding_serialized = pickle.dumps(query_video_embedding)
                    add_embeddings(video_embedding_serialized, None)
                else:
                    st.markdown(f'<p class="result">Это дубликат видео под ID: {similar_uuid},\
                    коэффициент сходства равен {video_similarity_rate} </p>',
                                unsafe_allow_html=True)
                    print(f'Query video is dublicate for uuid = {similar_uuid},\
                        video_similarity_rate = {video_similarity_rate}')
        
            else:
                st.markdown(f'<p class="result">Это дубликат видео под ID: {similar_uuid},\
                    коэффициент сходства равен {video_similarity_rate} </p>',
                                unsafe_allow_html=True)
                print(f'Query video is dublicate for uuid = {similar_uuid},\
                        video_similarity_rate = {video_similarity_rate}')
        else:
            video_embedding_serialized = pickle.dumps(query_video_embedding)
            add_embeddings(video_embedding_serialized, None)

    if os.path.exists(video_path):
        os.remove(video_path)