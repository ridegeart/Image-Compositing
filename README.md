# Image-Compositing
Implementation of compositing craters on airport runway.

## Build Rnuway Craters Datasets
- Craters Datsets
- Runway background
- Runway Craters Datasets

### Craters Datsets
1. Craters：dowload from robotflow
    - web：https://universe.roboflow.com/rdd-jqqq8/bomb-craters-low/dataset/1
    - google：https://universe.roboflow.com/rdd-jqqq8/google_earth/dataset/1
    - Zip at `./datasets`
2. Craters prepocessing (source of craters)
    - web：`./datasets/web_crater.py`  
        - Perspective Transform
        - Remove Background
        - Image Augmentation
        - parameter settings：
            1. `src_path`：craters images path load from Robotflow
            2. `gnd_path`：craters labels path load from Robotflow
            3. `dst_path`：Path to save perspective transformed images-default  `images_seg`
            4. `remove_bg_path`：Path to save Removed Background images-default  `images_remove_bg`
    - google：：`./datasets/google_crater.py`  
        - Segmentation craters
        - Build crater mask
        - parameter settings：
            1. `src_path`：craters images path load from Robotflow
            2. `gnd_path`：craters labels path load from Robotflow
            3. `dst_path`：Path to save craters after segmentation-default  `images_seg`
            4. `mask_path`：Path to save mask of craters-default  `mask`
    ```
        .
        ├── datasets
        │   ├── web
        │   │   ├── train
        │   │   │  ├── images_seg
        │   │   │  └── images_remove_bg
        │   │   └── test
        │   └── google
        │       ├── train
        │       │  ├── images_seg
        │       │  └── mask
        │       └── test
    ```

### Runway background

| itri  | itri_small  | itri_shadow  | airport_runway  |
| ---------- | -----------| -----------| -----------|
| Origin Road   | Resize fit GSD=16.7   | Road with significant tree shadow   | serching from web   |
| 21039(w) * 1561(h)   | 1256 * 95   | 15785 * 1561   | 3840 * 2160   |
| runway-top-view_raw.tif   | image_airport.jpg   | 20240420_bg3.jpg   | Airport-plane-runway-top-view_3840x2160.jpg   |
|    | ![image](https://github.com/ridegeart/Image-Compositing/blob/main/datasets/image_airport.jpg)   | ![image](https://github.com/ridegeart/Image-Compositing/blob/main/datasets/20240420_bg3.jpg)   | ![Airport-plane-runway-top-view_3840x2160](https://github.com/ridegeart/Image-Compositing/assets/73794853/b55af0ea-048b-4f3d-95fe-3b1fad34e1c6)
   |

### Full-Runway Craters Datasets
Gene craters(fg) onto runway background.  
1. Create mask of fg
2. Cropped ROI on bg
3. Cutout fg mask on ROI of bg
4. Add 1. and 3.
5. Put the composited ROI onto origin runway bg

- web：`./datasets/image_synthesis_web.py`
    - Gene train/test Full-Runway + web craters datasets.
    - Setting train：1000 images / test：200 images.
    - `dst_path`：fusion images saved path  -  default `./{mod}_{bg_src}/images_whole`  
    If mod = train， bg_src = itri，dst_path = ./train_itri/images_whole
    - `crater_dir`：craters source - default `web`
    - `bg_remove`：prepeocessed craters path - default `images_remove_bg`
    - parameter settings：
        1. `bg_src`：choose between : itri/itri_shadow/airport_runway.
        2. `GSD`：resize craters size according to bg's GSD.   
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
    - Gene train/test Full-Runway + google craters datasets.
    - Setting train：1000 images / test：200 images.
    - `dst_path`：fusion images saved path - default `./{mod}_{bg_src}/images_whole`
    - `crater_dir`：craters source - default `google`  
    - `bg_remove`：prepeocessed craters path - default `images_seg`
    - parameter settings：
        1. `bg_src`：choose between : itri/itri_small/airport_runway

## Gene Datasets
`./datasets/gene_datasets.py`  An ensembled file to gene train/test datasets. Incliuding following moduls  :
- Gene Full-Runway Craters name list
- Trans Yolo format to VOC
- Sliding windows
- parameter settings：
    1. `bg_src`：choose between : itri/itri_shadow/airport_runway
    2. `sliding_window`：True / Flase , gene cropped images or not

### Sliding windows  
Performing detection over smaller slices of the Full-Runway Craters image and then merging the sliced predictions on the original image.
- sliding windows overlap = 50% means stridex/stridey length is half of windows width/windows height.
- calculate crater area inside the windows if intersect area > 70%, build label in this windows.
- `windows_width`：sliding windows width - defalut `3000` 
- `windows_height`：sliding windows height - defalut `1220` 
- `imgpath`：Full-Runway Craters images path-refer to **Runway Craters Datasets**
- `srcAnn`：Full-Runway Craters labels path-refer to **Runway Craters Datasets**
- `annotation`：Full-Runway Craters images name list - defalut `./{mod}_{bg_src}/{mod}.txt` 
- `cropAnno`：sliced images save path - defalut `images` 
- `savePath`：sliced labels save path - defalut `labels` 
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
Trainning with mmdetection with pretrainned swin-transformer model.
### Build Environment 

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
| method     | Input/output  | command  | format  |
| ---------- | -----------| -----------| -----------| 
| mmdetection built-in    | pkl   | all images result  | [class_id,VOC,confidence]  |
| Inference file   | txt   | each images one files   | [class_id,YOLO] | 
- input images size：`./mmdetection/configs/_base_/datasets/crater_w12_3.py`, defult = (720, 720)
### mmdetection built-in 
- test config settings：`./mmdetection/configs/_base_/datasets/crater_w12_3.py`
    ```
    cd mmdetection  
    python tools/test.py configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py ./work_dirs/exp6/last.pth --out results_mmdet/results_val.pkl
    ```  
    if trainning with sliding windows , merging the sliced predictions to the original image than analysis the mAP.
- merge predict result ：`./datasets/joint_image_pickle.py`
    - calculate bbox_iou, delete objects overlap > 50%
    - .pkl file each object format：[xmin, ymin, xmax, ymax, confidence]
    - parameter settings：
        1. `big_images_path`：Full-Runway Craters images path
        2. `cropped_name_list`：sliced images name list path
        3. `joint_name_list`： Full-Runway Craters images name list path
        4. `result_path`：.pkl file path return by test.py
        5. `joint_path`：saved .pkl file joint from result pkl file

### Inference file
1. Inference Full-Runway Craters images
    - Including following moduls  :
        - Sliding windows  
        - Predict
        - Merge sliced predictions

    - `./mmdetection/inference_test.py`
        ```
        cd mmdetection  
        python inference_test.py configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py ./work_dirs/exp6/latest.pth /home/training/datasets/fusion_image/val/images_bg/ 0.5 0.5
        ``` 
    - parameter settings：
        1. `config`：test config file path
        2. `checkpoint`：checkpoint file path
        3. `source`：test images path (images_whole)
        4. `conf_thres`：test confidence threshold
        5. `iou_thres`：test iou threshold
2. Inference sliced images
    - `./mmdetection/inference.py`
        ```
        cd mmdetection  
        python inference.py configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py ./work_dirs/exp6/latest.pth /home/training/datasets/fusion_image/test_itri/images/ 0.5 0.5
        ``` 

## Model Analysis
1. mAP
    - use pkl files return by `tools/test.py`
        ```
        cd mmdetection  
        python ./tools/analysis_tools/eval_metric.py ./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py ./results_mmdet/results_itri_bg.pkl --eval mAP
        ```
    - If trainning with sliced images , want to calculate mAP with merged pkl file：
        - parameter settings：`./mmdetection/configs/_base_/datasets/crater_w12_3.py`
            1. test.dataroot.ann_file：Full-Runway Craters images name list path
            2. test.dataroot.img_prefix：Full-Runway Craters images path
        - `./datasets/test_XXX/label_xml`：should be Full-Runway Craters labels

2. accuracy of different size boxes
    - `tools/all_box.py` ： summarize nums of all size
    - `tools/error_box.py` ： missed detect box
    - parameter settings：
        1. `gt_folder` ： 
        2. `pred_folder` ： 
        3. `width`/`height` ： input images size

3. draw bboxes
    - `tools/draw_box.py`
