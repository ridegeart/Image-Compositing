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
        1. `src_path`：Origin craters images from Robotflow
        2. `gnd_path`：Origin craters images from Robotflow
        3. `dst_path`：Path to save perspective transformed images
        4. `remove_bg_path`：Path to save Removed Background images
    - google：：`./datasets/web_craters.py`  
        - Get craters by read segmentation label
        - Use getCounters to get crater mask
        1. `src_path`：Origin craters images from Robotflow
        2. `gnd_path`：Origin craters images from Robotflow
        3. `dst_path`：Path to save craters after segmentation
        4. `mask_path`：Path to save mask of craters
### Runway background

| itri  | itri_small  | itri_shadow  | airport_runway  |
| ---------- | -----------| -----------| -----------|
| Origin Road   | Lower size   | Road with significant tree shadow   | serching from web   |
| 21039(w) * 1561(h)   | 1256 * 95   | 15785 * 1561   | 3840 * 2160   |

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
    1. `dst_path`：Path to save images
    2. `crater_dir`：Path where craters saved - set at `web`
    3. `bg_remove`：Path of bg removed craters saved
    4. `bg_src`：choose from : itri/itri_shadow/airport_runway
    5. `GSD`：Decide craters size from bg's GSD.   
    If 1pix = 1cm (real) , GSD = 1； 1pix = 1.6cm (real) , GSD = 1.6
- google：`./datasets/image_synthesis_google.py`  
    1. `crater_dir`：Path where craters saved - set at `google`  
    2. `bg_src`：choose from : itri/itri_small/airport_runway

## Gene Datasets
`./datasets/gene_datasets.py`  An ensembled file to gene train/test datasets. Incliuding following moduls  :
- gene images_whole list
- gene VOC from Yolo format
- Sliding windows on big pic
1. `bg_src`：choose from : itri/itri_shadow/airport_runway
2. `sliding_window`：True or Flase , making cropped images or not

### Sliding windows



