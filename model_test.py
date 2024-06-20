#coding=utf-8
 
from re import L
from mmdetection.mmdet.apis import init_detector
from mmdetection.mmdet.apis import inference_detector
import numpy as np
from PIL import Image
from mmcv import Config, DictAction
from datasets.image_cropped_test import crop_dataset
from datasets.joint_image import joint_main, output_result
import glob
import tqdm
import os
import argparse

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x,y,w,h]

#convert minX,minY,maxX,maxY to normalized numbers required by Yolo
def getYoloNumbers(imagePath, minX,minY,maxX, maxY):
    image=Image.open(imagePath)
    w= int(image.size[0])
    h= int(image.size[1])
    b = (minX,maxX, minY, maxY)
    bb = convert((w,h), b)
    image.close()
    return bb

def run_model():
    config_file = './mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py'
    checkpoint_file = './mmdetection/checkpoint/epoch_4.pth'
    conf_thres = 0.5
    iou_thres = 0.5

    cfg = Config.fromfile(config_file)
    #cfg['test_pipeline'][1]['img_scale'] = (2852, 1220)
    cfg.model.test_cfg.rcnn.score_thr = float(conf_thres)
    cfg.model.test_cfg.rcnn.nms.iou_threshold = float(iou_thres)

    model = init_detector(cfg, checkpoint_file, device='cuda:0')
    return model
    
def predict(path):
    #load models
    model = run_model()

    # cropped input image
    windows_width = 3000
    windows_height = 1220

    image_name = os.path.basename(path)
    print(image_name)

    if not os.path.exists(path) or not image_name.endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
        return output_result(None,path)['status']
    else:
        data = output_result(None,path)
        data['status']["message"] = "Get image successfully."
        data['status']["code"] = 1
        print(data['status'])
        
    rawImg_path = path.replace('/'+image_name, '')
    rawImg_dir = os.path.basename(rawImg_path)
    base_path = path.replace(rawImg_dir+'/'+image_name, '')
    savePath = base_path + 'images_test'

    crop_dataset(path, windows_width, windows_height, savePath)

    # predict
    i = 0
    subdir = "./mmdetection/runs"
    filepath = subdir + "/exp"
    while os.path.exists(filepath):
        i += 1
        folder = f'/exp{i}'
        filepath = subdir+folder
    txtpath = filepath+"/labels"
    os.makedirs(txtpath)

    for image_name in tqdm.tqdm(os.listdir(savePath)): #crop_dataset saved image path
        imgpath = os.path.join(savePath, image_name)
        result = inference_detector(model, imgpath)
        bbox_result = result
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        bbox_str = ""
        for bbox, label in zip(bboxes, labels):
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])
            label = int(label)
            [x,y,w,h] = getYoloNumbers(imgpath,xmin,ymin,xmax, ymax)
            # if bbox[4] > 0.2:
            bbox_str += str(label) + ' ' + str(x) + ' ' \
                        + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
        # with open('results_mmdet/detectors_22_1_e20_88_iou0.9999_conf0.50.txt', 'a') as w:
        with open(txtpath+"/"+image_name[:-4] + ".txt", "w") as f:
            f.write(bbox_str)
    
    #joint
    yolo_labels_path = os.path.join(filepath,'labels_croped_fusion')
    result = joint_main(rawImg_path, txtpath, yolo_labels_path, windows_width, windows_height)
    return result
    


if __name__ == '__main__':
    path = '/home/training/datasets/fusion_image/val/images_bg/20240420_bg.jpg'
    predict(path)