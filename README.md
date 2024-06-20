# MMdetection on Flask
Implement ML model on web server through Flask.

## mmdet_api architecture
```
./mmdet_api
├── datasets
├── mmdetection
│   └── checkpoint
│        └── epoch_4.pth
├── model.py
├── requirements.txt
├── environment.yml
├── run.py
└── test.py
```

## Environment Build
#copy and build new environment
```
conda env create -f environment.yml
```
#cuda >= 11.1
```
pip install -r requirements.txt
```
#mmdetection v2.25.0
```
cd mmdetection  
pip install -r requirements/build.txt  
pip install -v -e .  
```

## Package predict model (model.py)
```python
#載入模型
class mmdet(object):
    def __init__(self, config_file = './mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py', 
                checkpoint_file = './mmdetection/checkpoint/epoch_4.pth', 
                conf_thres = 0.5, iou_thres = 0.5):
    ...

# 模型預測的function
def predict(path):
    #load models
    model = mmdet()
    ...
```

## Flask API (run.py)
```python
import model
@app.route('/pothole_recognize', methods=['POST'])
def postInput():
    # 取得前端傳過來的數值
    values = request.form
    img_path = values['image_path']
    # 建立模型，預測並返回結果
    result = model.predict(img_path)
    print(result)

    return jsonify(result)
```
## Example (test.py)
```python
import requests
import json

req = requests.post('http://127.0.0.1:5000/pothole_recognize', data = {'image_path':'/home/training/datasets/fusion_image/val/images_bg/20240420_bg.jpg'})
```