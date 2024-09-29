import librosa
import pickle
from moviepy.editor import VideoFileClip


def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


def extract_audio_embedding(audio):
    model = load_model('huysosi.onnix')
    audio_data, _ = librosa.load(audio[0], sr=48000)
    audio_data = audio_data.reshape(1, -1)
    audio_embed = model.get_audio_embedding_from_data(x=audio_data, use_tensor=False)
    return audio_embed


def extract_video_embedding(video):
    model = load_model('huysosi.onnix')
    embedding = model.make_embedding(video)  # Тут как бы не то
    return embedding


def create_audio(video):
    video = VideoFileClip(video)
    audio = video.audio # Как wav файл сохранить не в таблице а в ОЗУ
    return audio
