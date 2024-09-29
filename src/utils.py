import os
import requests

def download_file_from_s3(bucket_url, object_key, download_path):
    directory = os.path.dirname(download_path)
    if not os.path.exists(directory):
        dir_path = os.path.join(os.getcwd(), directory)
        os.makedirs(dir_path)
        
    object_url = os.path.join(bucket_url, object_key)

    response = requests.get(object_url)

    if response.status_code == 200:
        with open(download_path, 'wb') as file:
            file.write(response.content)
        print("Модель успешно скачана и сохранена в:", download_path)
    else:
        print("Ошибка при скачивании модели:", response.status_code)


def mp4(name):
    return name+'.mp4'


def compute_similarities(q_feat, d_feat, topk_cs=True):
    sim = q_feat @ d_feat.T
    sim = sim.max(dim=1)[0]
    if topk_cs:
        sim = sim.sort()[0][-3:]
    sim = sim.mean().item()
    return sim

def calculate_similarities(video_id=None, second_video_id=None, all_features=None):
    if video_id is not None and second_video_id is not None:
        return compute_similarities(all_features[video_id], all_features[second_video_id])
    
    elif video_id is not None and all_features is not None:
        first_feat = all_features[video_id]
        similarities = {second_video_id: compute_similarities(first_feat, all_features[second_video_id]) 
                        for second_video_id in all_features if second_video_id != video_id}
        return similarities
    
    elif all_features is not None:
        similarities = {video_id: compute_similarities(video_id, all_features=all_features) 
                        for video_id in all_features}
        return similarities

    return None  # В случае, если не указаны необходимые переменные