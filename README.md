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
- web：`./datasets/image_synthesis_web.py`
    - Gene craters(fg) onto runway background
    - Use getCounters to get crater mask
- google：`./datasets/image_synthesis_google.py`


