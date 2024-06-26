#coding=utf-8
 
from re import L
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
import numpy as np
from PIL import Image
from mmcv import Config, DictAction
from image_cropped_test import crop_dataset
from joint_image import joint_main
import glob
import tqdm
import os
import argparse


parser = argparse.ArgumentParser(description='...')
parser.add_argument('config', help='test config file path')
parser.add_argument('checkpoint', help='checkpoint file')
parser.add_argument('source', help='test images path')
#parser.add_argument('imgsw', help='test img width')
#parser.add_argument('imgsh', help='test img height')
parser.add_argument('conf_thres', help='test imgsz')
parser.add_argument('iou_thres', help='test imgsz')
args = parser.parse_args()


config_file = args.config
checkpoint_file = args.checkpoint
path = args.source
#imgsw = args.imgsw
#imgsh = args.imgsh
conf_thres = args.conf_thres
iou_thres = args.iou_thres

cfg = Config.fromfile(config_file)
#cfg['test_pipeline'][1]['img_scale'] = (imgsw, imgsh)
cfg.model.test_cfg.rcnn.score_thr = float(conf_thres)
cfg.model.test_cfg.rcnn.nms.iou_threshold = float(iou_thres)

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

def predict(path):

    # cropped input image
    windows_width = 3000
    windows_height = 1220

    for image_name in tqdm.tqdm(os.listdir(path)):
        print(image_name)
        rawImg_dir = os.path.basename(path)
        base_path = path.replace(rawImg_dir, '')
        annotation = base_path + 'val.txt'
        savePath = base_path + 'images_test'
        imgpath = os.path.join(path, image_name)

        crop_dataset(imgpath, windows_width, windows_height, annotation, savePath)

    # predict
    i = 0
    subdir = "./runs"
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
    joint_main(path, txtpath, yolo_labels_path, windows_width, windows_height,
               scale=1)

if __name__ == '__main__':
    model = init_detector(cfg, checkpoint_file, device='cuda:0')
    predict(path)