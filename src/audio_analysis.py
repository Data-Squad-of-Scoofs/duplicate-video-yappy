import torch
import librosa
from moviepy.editor import VideoFileClip
import numpy as np


def load_and_preprocess_audio(video_path,
                              fps=48000):
    """Функция, в которой подгружается и обрабатывается аудио"""

    video = VideoFileClip(video_path)
    audio = video.audio

    audio_data = audio.to_soundarray(fps=fps)
    audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data.reshape(1, -1)

    return audio_data


def get_audio_features(model,
                       audio_data):

    with torch.no_grad():
        audio_embed = model.get_audio_embedding_from_data(
            x=audio_data, use_tensor=False)

    audio_embed = torch.tensor(audio_embed)

    return audio_embed


def get_audio_tempo(audio_data):
    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=48000)

    tempo = torch.tensor(tempo)

    return tempo
