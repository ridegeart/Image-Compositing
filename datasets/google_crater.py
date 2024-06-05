import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os

def load_data(ground_truth_xml):
  
  ground_truth = {}
  gnd_bboxes = {}
  truth_polygon = []
  file = os.path.basename(ground_truth_xml)
  tree = ET.parse(ground_truth_xml)
  root = tree.getroot()
  
  for obj in root.findall('object'):
    bnd_box = obj.find('bndbox')
    polygons = obj.find('polygon')

    x = {}
    y = {}
    
    for i in range(int(len(polygons)/2)):
        key_x = 'x{}'.format(i+1)
        key_y = 'y{}'.format(i+1)
        
        x = int(float(polygons.find(key_x).text))
        y = int(float(polygons.find(key_y).text))
        truth_polygon.append((x,y))
        
    xmin = int(bnd_box.find('xmin').text)
    ymin = int(bnd_box.find('ymin').text)
    xmax = int(bnd_box.find('xmax').text)
    ymax = int(bnd_box.find('ymax').text)
    
    if file not in ground_truth:
      ground_truth[file] = [truth_polygon]
      gnd_bboxes[file] = [[xmin, ymin, xmax, ymax]]
    else:
      ground_truth[file].append(truth_polygon)
      gnd_bboxes[file].append([xmin, ymin, xmax, ymax])
    
    truth_polygon = []

  return ground_truth, gnd_bboxes

def cropped_object(name, gnd_truth, gnd_bboxes):

  if os.path.isfile(src_path):
    image = cv2.imread(os.path.join(src_path))
  else:
    image = cv2.imread(os.path.join(src_path,name))
  img =  255 * np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
  file_name = name[:-4]
  
  contours = gnd_truth[name[:-4]+'.xml']
  bboxes = gnd_bboxes[name[:-4]+'.xml']
  
  for i,contour in enumerate(contours):
    contour = np.array(contour)
    xmin, ymin, xmax, ymax = bboxes[i]
    cv2.drawContours(img, [contour.reshape(-1,1,2)], -1, (255, 255, 255), -1)
    bg = cv2.bitwise_and(image, img)
    cv2.imwrite(os.path.join(dst_path,'{}_{}.jpg'.format(file_name,i)), bg[ymin:ymax+1,xmin:xmax+1,:])
    cv2.imwrite(os.path.join(mask_path,'{}_{}.jpg'.format(file_name,i)), img[ymin:ymax+1,xmin:xmax+1,:])
 
if __name__ == '__main__': 

  src_path = "./google_earth-1/train/images"
  gnd_path = "./google_earth-1/train/labels/" #endwith '/'
  dst_path = "./google_earth-1/train/images_seg2"
  mask_path = "./google_earth-1/train/mask1"
  
  if not os.path.exists(mask_path):
    os.mkdir(mask_path)
  
  if not os.path.exists(dst_path):
    os.mkdir(dst_path)
    
  if os.path.isfile(src_path):
    f = os.path.basename(src_path)
    gnd_truth, gnd_bboxes = load_data(gnd_path+f[:-4]+'.xml')
    cropped_object(f, gnd_truth, gnd_bboxes)
    
  elif os.path.isdir(src_path):
    for f in os.listdir(src_path):
      gnd_truth, gnd_bboxes = load_data(gnd_path+f[:-4]+'.xml')
      cropped_object(f, gnd_truth, gnd_bboxes)