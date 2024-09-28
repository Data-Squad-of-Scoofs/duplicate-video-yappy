import torch
from tqdm import tqdm
import os
from .utils import compute_similarities

def calculate_similarities(video_id=None, second_video_id=None, all_features=None, model=None):
    if video_id is not None and second_video_id is not None:
        return compute_similarities(all_features[video_id], all_features[second_video_id])
    
    elif video_id is not None and all_features is not None:
        first_feat = all_features[video_id]
        similarities = {second_video_id: compute_similarities(first_feat, all_features[second_video_id]) 
                        for second_video_id in all_features if second_video_id != video_id}
        return similarities
    
    elif all_features is not None:
        similarities = {video_id: compute_similarities(video_id, all_features=all_features, model=model) 
                        for video_id in all_features}
        return similarities

    return None  # В случае, если не указаны необходимые переменные


def get_video_features(preprocess, model, features=None, video_folder=None, video_paths=None):
    if video_folder is not None:
        video_paths = [os.path.join(video_folder, name) for name in os.listdir(video_folder)]
    elif video_paths is None:
        raise ValueError("Either video_folder or video_paths must be provided")

    print("Starting to extract video features....")

    if features:
        all_features = features
    else:
        all_features = {}

    for path in tqdm(video_paths, ncols=70):
        video_id = os.path.basename(path.split('.')[0])

        if video_id in all_features:
            continue   

        frames = torch.from_numpy(preprocess(path)).permute(0, 4, 1, 2, 3).float().cuda()

        frames = frames.cuda()
        with torch.no_grad():
            feats = model.extract_features(frames)
        all_features[video_id] = feats.detach().cpu()

    normed_features = {vid: model.normalize_features(feats.cuda()) for vid, feats in all_features.items()}

    return normed_features
