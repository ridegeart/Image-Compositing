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

## Environment Build-cuda 11.1
#copy and build new environment
```
conda env create -f environment-cuda111.yml
```
#cuda >= 11.1
```
pip install -r requirements-torch1.8.txt
```
#mmdetection v2.25.0
```
cd mmdetection  
pip install -r requirements/build.txt  
pip install -v -e .  
```

## Environment Build-cuda 11.8
#copy and build new environment
```
conda env create -f environment.yml
```
#cuda == 11.8
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```
#mmdetection
```
pip install openmim    
mim install mmcv-full
mim install mmcls
pip install mmcv==1.6.0
```
#Build mmdetection
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
def predict(model,path):

    ...
```

## Flask API (run.py)
```python
import model

# 載入模型
mmdet_model = model.mmdet()

@app.route('/pothole_recognize', methods=['POST'])
def postInput():
    # 取得前端傳過來的數值
    values = request.form
    img_path = values['image_path']
    # 預測並返回結果
    result = model.predict(mmdet_model, img_path)

    return jsonify(result)
```
## Example (test.py)
```python
import requests
import json

req = requests.post('http://127.0.0.1:5000/pothole_recognize', data = {'image_path':'/home/training/datasets/fusion_image/val/images_bg/20240420_bg.jpg'})
```