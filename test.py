import requests
import json


req = requests.post('http://127.0.0.1:5000/pothole_recognize', data = {'image_path':'/home/training/datasets/fusion_image/val/images_bg/20240420_bg.jpg'})
print(req.text)