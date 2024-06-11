#coding=utf-8
 
from re import L
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from mmcv import Config, DictAction
import glob
import tqdm
import os
import argparse


parser = argparse.ArgumentParser(description='...')
parser.add_argument('config', help='test config file path')
parser.add_argument('checkpoint', help='checkpoint file')
parser.add_argument('source', help='test images path')
parser.add_argument('imgsz', help='test imgsz')
parser.add_argument('conf_thres', help='test imgsz')
parser.add_argument('iou_thres', help='test imgsz')
args = parser.parse_args()


config_file = args.config
checkpoint_file = args.checkpoint
path = args.source
imgsz = args.imgsz
conf_thres = args.conf_thres
iou_thres = args.iou_thres

cfg = Config.fromfile(config_file)
cfg['test_pipeline'][1]['img_scale'] = (imgsz, imgsz)
cfg.model.test_cfg.rcnn.score_thr = float(conf_thres)
cfg.model.test_cfg.rcnn.nms.iou_threshold = float(iou_thres)

model = init_detector(cfg, checkpoint_file, device='cuda:0')

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

i = 0
subdir = "./runs"
filepath = subdir + "/exp"
while os.path.exists(filepath):
  i+=1
  folder = f'/exp{i}'
  filepath = subdir+folder
txtpath = filepath+"/labels"
os.makedirs(txtpath)
  
result = inference_detector(model, path)
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
  [x,y,w,h] = getYoloNumbers(path,xmin,ymin,xmax, ymax)
  # if bbox[4] > 0.2:
  bbox_str += str(label) + ' ' + str(x) + ' ' \
              + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
# with open('results_mmdet/detectors_22_1_e20_88_iou0.9999_conf0.50.txt', 'a') as w:
with open(txtpath+"/"+ "2024" + ".txt", "w") as f:
  f.write(bbox_str)