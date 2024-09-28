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