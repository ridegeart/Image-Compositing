import requests
import json


req = requests.post('http://127.0.0.1:5000/predict', data = {'image_path':'/home/training/datasets/fusion_image/val/images_bg'})
print(req.text)