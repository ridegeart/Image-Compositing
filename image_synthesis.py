import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image,ImageDraw,ImageFont
from bg_preprocessing import RandBackground

def load_image(image_path):
    crater_list = os.listdir(image_path)
    rnd_index = random.randint(0,len(crater_list)-1)
    f = crater_list[rnd_index]
    image_crater = cv2.imread(os.path.join(image_path,f), cv2.IMREAD_UNCHANGED)
    image_crater,[min_x,min_y,max_x,max_y] = image_preprocessing(image_crater)
    #kernel = np.ones((1,1), np.uint8)
    #image_crater = cv2.erode(image_crater, kernel, iterations = 1)
    return image_crater[min_y:max_y+1,min_x:max_x+1]
    
def image_preprocessing(image):
  mask = image[:,:,3]
  
  h = image.shape[0]
  w = image.shape[1]
  max_y = 0
  max_x = 0
  min_y = h
  min_x = w
  
  for x in range(w):
      for y in range(h):
          if mask[y, x]<1:
              image[y, x] = [0,0,0,255]
          else:
            max_y = max(y, max_y)
            max_x = max(x, max_x)
            min_x = min(x, min_x)
            min_y = min(y, min_y)
  return image[:,:,:3], (min_x,min_y,max_x,max_y)
  
# resize image by GSD
def resize_img(image):
    (h, w, d) = image.shape
    max_side = h if (h > w) else w
    min_size = 16
    max_size = 516
    img_fx_min = min_size/max_side
    img_fx_max = max_size/max_side
    size = random.uniform(img_fx_min, img_fx_max)
    return cv2.resize(image, dsize=None, fx=size, fy=size)

# 旋轉圖片
def rotate_img(image):
    (h, w, d) = image.shape
    center = (w // 2, h // 2)
    angle = random.randint(-40, 40) / 20
    M = cv2.getRotationMatrix2D(center, angle, 1.04)
    image = cv2.warpAffine(image, M, (w, h))
    return image
    
def image_mask(image):
    crater2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(crater2gray, 1, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))# 開運算去除mask中白色雜訊
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #mask = cv2.GaussianBlur(mask, (9, 9), 0)# 去除mask的黑色邊框
    return mask

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


def label_save(file, text):
    lable_dir = "./fusion_image/train/labels1"
    txt_file = os.path.join(lable_dir,file)
    with open(txt_file, 'w') as f:
        f.writelines(text) 

def fusion_image (i, image, label_text):

  if i <= 0:
    return image, label_text
    
  else:    
    # Read craters image
    crater_path = "./Bomb-craters-low-1/train/images_remove_bg"
    
    image_crater = load_image(crater_path)
    
    # Loop until a non-overlapping location is found
    while True:
      # random crater size
      resized_image = resize_img(image_crater)
      cols, rows = resized_image.shape[:2]
      resized_image = rotate_img(resized_image)
      # Random location with upper bound check
      y = random.randint(0, image.shape[0] - cols-1)
      x = random.randint(0, image.shape[1] - rows-1)
  
      # Check for overlap with existing craters (assuming craters are opaque)
      roi = image[y:y+cols, x:x+rows]
      is_overlapping = np.any(roi != airport_copy[y:y+cols, x:x+rows])  # Check if any pixel is different
      if not is_overlapping:
        break
    
    # 產生去背圖像的mask
    crater_copy = resized_image.copy()
    mask = image_mask(crater_copy)
        
    # 背景上ROI區域摳出圖案mask
    mask_inv = cv2.bitwise_not(mask)
    airport_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    
    # 將「摳出圖案mask的背景」與「填充圖案的前景」相加
    dst = cv2.add(airport_bg, resized_image)
    
    #image = cv2.seamlessClone(resized_image, image, mask, (x+rows//2, y+cols//2), cv2.NORMAL_CLONE)

    # 用dst替換掉背景中含有彈坑的區域
    image[y:y+cols, x:x+rows] = dst
    
    # 儲存bounding box
    h= int(image.shape[0])
    w= int(image.shape[1])
    [x,y,w,h] = convert((w,h), (x, x+rows, y, y+cols))
    label = "0"+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h)+"\n"
    label_text.append(label)
    
    return fusion_image (i-1, image, label_text)

if __name__ == '__main__': 
  
  fusion_image_number = 1000
  
  # Fuse images
  for i in range(fusion_image_number):
    # Random number of craters
    num_craters = random.randint(1, 10)  
    # Read background image
    bg_path = 'Airport-plane-runway-top-view_3840x2160.jpg'
    image_airport = cv2.imread(bg_path)
    airport_copy = image_airport.copy()
    
    label=[]
    dst, label_text = fusion_image(num_craters, image_airport, label)
    # saved images
    cv2.imwrite(f"./fusion_image/train/images1/image_crater{i}.jpg", dst)
    # saved labels
    label_save(f"image_crater{i}.txt",label_text)
    print(f"saved_{i}")
  