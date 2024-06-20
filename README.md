# Image-Compositing
Implementation of compositing craters on airport runway.

## Rnuway Craters Datasets
- Craters Datsets
- Runway background
- Runway Craters Datasets

### Craters Datsets
1. Craters：dowload from robotflow
    - web：https://universe.roboflow.com/rdd-jqqq8/bomb-craters-low/dataset/1
    - google：https://universe.roboflow.com/rdd-jqqq8/google_earth/dataset/1
    - Zipped at `./datasets`
2. Craters prepocessing (source of craters)
    - web：`./datasets/web_craters.py`  
        - Perspective Transform
        - Remove Background
        - Image Augmentation
        - parameter settings：
            1. `src_path`：Origin craters images from Robotflow
            2. `gnd_path`：Origin craters images from Robotflow
            3. `dst_path`：Path to save perspective transformed images-default  `images_seg`
            4. `remove_bg_path`：Path to save Removed Background images-default  `images_remove_bg`
    - google：：`./datasets/web_craters.py`  
        - Get craters by read segmentation label
        - Use getCounters to get crater mask
        - parameter settings：
            1. `src_path`：Origin craters images from Robotflow
            2. `gnd_path`：Origin craters images from Robotflow
            3. `dst_path`：Path to save craters after segmentation-default  `images_seg`
            4. `mask_path`：Path to save mask of craters-default  `mask`
    ```
        .
        ├── datasets
        │   └── web
        │       ├── train
        │       │  ├── images_seg
        │       │  └── images_remove_bg
        │       └── test
    ```

### Runway background

| itri  | itri_small  | itri_shadow  | airport_runway  |
| ---------- | -----------| -----------| -----------|
| Origin Road   | Lower size   | Road with significant tree shadow   | serching from web   |
| 21039(w) * 1561(h)   | 1256 * 95   | 15785 * 1561   | 3840 * 2160   |
| runway-top-view_raw.tif   | image_airport.jpg   | 20240420_bg3.jpg   | Airport-plane-runway-top-view_3840x2160.jpg   |

### Runway Craters Datasets
Gene craters(fg) onto runway background.  
1. Create mask of fg
2. Cropped ROI of bg
3. Cutout fg mask on ROI of bg
4. Add 1. and 3.
5. Put the composited ROI onto origin runway

- web：`./datasets/image_synthesis_web.py`
    - gene train/test images_whole with full length origin pic
    - setting 1000 of trainning and 200 of testing
    - `dst_path`：Path to save images-default `./{mod}_{bg_src}/images_whole`  
    If mod = train， bg_src = itri，dst_path = ./train_itri/images_whole
    - `crater_dir`：Path where craters saved - default `web`
    - `bg_remove`：Path of bg removed craters saved - default `images_remove_bg`
    - parameter settings：
        1. `bg_src`：choose from : itri/itri_shadow/airport_runway
        2. `GSD`：Decide craters size from bg's GSD.   
    If 1pix = 1cm (real) , GSD = 1； 1pix = 1.6cm (real) , GSD = 1.6
    ```
            .
            ├── datasets
            │   ├── web
            │   └── fusion_images
            │       ├── train_itri
            │       │  ├── images_whole
            │       │  └── labels_whole
            │       └── test_itri
    ```
- google：`./datasets/image_synthesis_google.py`  
    - same as web except crater source
    - `dst_path`：Path to save images-default `./{mod}_{bg_src}/images_whole`
    - `crater_dir`：Path where craters saved - default `google`  
    - `bg_remove`：Path of bg removed craters saved - default `images_seg`
    - parameter settings：
        1. `bg_src`：choose from : itri/itri_small/airport_runway

## Gene Datasets
`./datasets/gene_datasets.py`  An ensembled file to gene train/test datasets. Incliuding following moduls  :
- gene images_whole list
- gene VOC from Yolo format
- Sliding windows on big pic
- parameter settings：
    1. `bg_src`：choose from : itri/itri_shadow/airport_runway
    2. `sliding_window`：True or Flase , making cropped images or not

### Sliding windows  
Using sliding windows with overlap = 50% to crop big runway craters images to small size images. length of stridex/stridey is half of kernelw/kernelh.
- calculate crater area inside this windows if intersect area > 70%, label in this windows
- `windows_width`：sliding windows size - defalut `3000` 
- `windows_height`：sliding windows size - defalut `1220` 
- `imgpath`：big runway craters images path-refer to **Runway Craters Datasets**
- `srcAnn`：big runway craters labels path-refer to **Runway Craters Datasets**
- `annotation`：big runway craters images name list - defalut `./{mod}_{bg_src}/{mod}.txt` 
- `cropAnno`：path saving cropped images - defalut `images` 
- `savePath`：path saving cropped labels - defalut `labels` 
    ```
            .
            ├── datasets
            │   ├── web
            │   └── fusion_images
            │       ├── train_itri
            │       │  ├── images_whole
            │       │  ├── images
            │       │  ├── labels_whole
            │       │  └── labels
            │       └── test_itri
    ```
## Model Trainning
Using mmdetection
### Environment Build

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
### Training config settings
`./mmdetection/configs/_base_/datasets/crater_w12_3.py`
- parameter settings：
    1. data_root : full path of datasets. ex：`D:/xx/xx/datasets/`
    2. data.ann_file/img_prefix data_root： train/test/val images name list + dir
- build work dir  ：  `./mmdetection/work_dirs/exp/`
### Training
```
cd mmdetection  
python ./tools/train.py ./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py --work-dir ./work_dirs/expX
```
## Model Inference
| method     | output  | command  | format  |
| ---------- | -----------| -----------| -----------| 
| mmdetection built-in    | pkl   | all images result  | VOC+Cofidence+class_id  |
| Inference file   | txt   | each images one files   | Yolo+class_id | 
- input images size：`./mmdetection/configs/_base_/datasets/crater_w12_3.py`, defult = (720, 720)
### mmdetection built-in 
- test config settings：`./mmdetection/configs/_base_/datasets/crater_w12_3.py`
    ```
    cd mmdetection  
    python tools/test.py configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py ./work_dirs/exp6/last.pth --out results_mmdet/results_val.pkl
    ```  
    if trainning with cropped images , should joint the cropped predict results than analysis the mAP.
- joint predict result ：`./datasets/joint_image_pickle.py`
    - calculate bbox_iou delete objects overlap > 50%
    - .pkl file each object format：[xmin, ymin, xmax, ymax, confidence]
    - parameter settings：
        1. `big_images_path`：big runway craters images path
        2. `cropped_name_list`：copped images name list path
        3. `joint_name_list`： big runway craters images name list path
        4. `result_path`：.pkl file path return by test.py
        5. `joint_path`：saved .pkl file joint from result pkl file

### Inference file
- Incliuding following moduls  :
    - Sliding windows  
    - predict
    - joint cropped predict results
- `./mmdetection/inference_test.py`
    ```
    cd mmdetection  
    python inference_test.py configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py ./work_dirs/exp6/last.pth /home/training/datasets/fusion_image/test/images/ 0.5 0.5
    ``` 
- parameter settings：
    1. `config`：test config file path
    2. `checkpoint`：checkpoint file path
    3. `source`：test images path (images_whole)
    4. `conf_thres`：test confidence threshold
    5. `iou_thres`：test iou threshold

## Model Analysis
1. mAP
    - use pkl files return by `tools/test.py`
        ```
        cd mmdetection  
        python ./tools/analysis_tools/eval_metric.py ./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py ./results_mmdet/results_itri_bg.pkl --eval mAP
        ```
2. accuracy of different size boxes
    - input xml files 
    - `tools/all_box.py` ： summarize nums of all size
    - `tools/error_box.py` ： missed detect box
