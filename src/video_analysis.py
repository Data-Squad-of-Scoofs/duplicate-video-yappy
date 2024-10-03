import torch
from .preprocess import Preprocess


def load_and_preprocess_video(video_path,
                              clip_len=8,
                              frame_interval=1,
                              channels=1):
    """Функция, в которой подгружается и обрабатывается видео"""

    preprocess = Preprocess(clip_len=clip_len, out_size=224,
                            frame_interval=frame_interval, channels=channels)

    frames = torch.from_numpy(preprocess(video_path))

    frames = frames.permute(0, 4, 1, 2, 3).float()

    return frames


def get_video_features(frames, model):
    """Функция, в которой выделяется признаки видео с помощью модели"""

    with torch.no_grad():
        feats = model.extract_features(frames.cuda())

    feats = feats.detach().cpu()

    normed_feats = model.normalize_features(feats)

    return normed_feats
