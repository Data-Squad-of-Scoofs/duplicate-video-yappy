from tqdm import tqdm
import torch
import librosa
import os
from moviepy.editor import VideoFileClip
import numpy as np


def get_audio_features(model,
                       features=None,
                       video_folder=None,
                       video_paths=None):
    if video_folder is not None:
        video_paths = [os.path.join(video_folder, name)
                       for name in os.listdir(video_folder)]
    elif video_paths is None:
        raise ValueError("Either video_folder or video_paths must be provided")

    print("Starting to extract audio features....")

    if features:
        all_features = features
    else:
        all_features = {}

    for path in tqdm(video_paths, ncols=70):
        video_id = os.path.basename(path.split('.')[0])

        if video_id in all_features:
            continue

        video = VideoFileClip(path)
        audio = video.audio

        if audio is None:
            continue

        audio_data = audio.to_soundarray(fps=48000)
        audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.reshape(1, -1)

        if audio is None:
            print(audio)

        with torch.no_grad():
            audio_embed = model.get_audio_embedding_from_data(
                x=audio_data, use_tensor=False)

        all_features[video_id] = torch.tensor(audio_embed)

    return all_features


def get_audio_features_tempo(features=None,
                             video_folder=None,
                             video_paths=None):
    if video_folder is not None:
        video_paths = [os.path.join(video_folder, name)
                       for name in os.listdir(video_folder)]
    elif video_paths is None:
        raise ValueError("Either video_folder or video_paths must be provided")

    print("Starting to extract audio features....")

    if features:
        all_features = features
    else:
        all_features = {}

    for path in tqdm(video_paths, ncols=70):
        video_id = os.path.basename(path.split('.')[0])

        if video_id in all_features:
            continue

        video = VideoFileClip(path)
        audio = video.audio

        if audio is None:
            continue

        audio_data = audio.to_soundarray(fps=44000)
        audio_data = np.mean(audio_data, axis=1)

        sr = 44000

        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
        # print(f'Tempo: {tempo} BPM')

        all_features[video_id] = torch.tensor(tempo)

    return all_features
