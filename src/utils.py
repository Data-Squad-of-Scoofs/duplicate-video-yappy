import os
import requests
import numpy as np
from tqdm import tqdm
import laion_clap

from models.model import SimilarityRecognizer


def download_file(object_url, download_path):
    directory = os.path.dirname(download_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    response = requests.get(object_url)

    if response.status_code == 200:
        with open(download_path, 'wb') as file:
            file.write(response.content)
        # print("Файл успешно скачан и сохранен в:", download_path)
    else:
        print("Ошибка при скачивании файла:", response.status_code)


def compute_similarity(q_feat, d_feat, topk_cs=True):
    sim = q_feat @ d_feat.T
    sim = sim.max(dim=1)[0]
    if topk_cs:
        sim = sim.sort()[0][-3:]
    sim = sim.mean().item()
    return sim


def get_video_model(device):
    video_model = SimilarityRecognizer(model_type="base", batch_size=8)
    video_model.to(device)
    video_model.load_pretrained_weights(
        "checkpoints/best_model_base_224_16x16_rgb.pth")
    video_model.eval()

    return video_model


def get_audio_model(device):
    audio_model = laion_clap.CLAP_Module(
        enable_fusion=False, device=device, amodel='HTSAT-base')
    audio_model.load_ckpt(
        'checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt')
    audio_model.eval()

    return audio_model


def find_most_similar_by_video(query_embedding, db_embeddings):
    max_similarity = float('-inf')
    max_similar_video_id = None
    for db_video_id, db_embedding, _ in db_embeddings:
        similarity = compute_similarity(query_embedding, db_embedding)

        if similarity > max_similarity:
            max_similarity = similarity
            max_similar_video_id = db_video_id

    return max_similar_video_id, max_similarity
