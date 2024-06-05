import cv2
import tqdm
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from bg_preprocessing import rotate_image,mask_road_region,convert,image_mask

min_y = 0
max_y = 0

def load_image(image_path, mask_path):
    crater_list = os.listdir(image_path)
    rnd_index = random.randint(0,len(crater_list)-1)
    f = crater_list[rnd_index]
    image_crater = cv2.imread(os.path.join(image_path,f))
    mask = cv2.imread(os.path.join(mask_path,f))
    return image_crater, mask

def fusion_image (i, image, label_text):

  if i <= 0:
    return image, label_text
    
  else:    
    # Read craters image
    image_crater, mask_crater = load_image(crater_path, mask_path)

    # Loop until a non-overlapping location is found
    while True:
      cols, rows = image_crater.shape[:2]
      resized_image = rotate_image(image_crater,None)
      mask = mask_crater.copy()
      # Random location with upper bound check
      if cols >= image.shape[0]:
        y = 0
        start = int((cols - image.shape[0])/2)
        resized_image = resized_image[start:start+image.shape[0],:,:]
        mask = mask[start:start+image.shape[0],:,:]
        cols, rows = resized_image.shape[:2]
      else:
        y = random.randint(1, image.shape[0] - cols)
      x = random.randint(1, image.shape[1] - rows-1)
      # Check for overlap with existing craters (assuming craters are opaque)
      roi = image[y:y+cols, x:x+rows]
      is_overlapping = np.any(roi != airport_copy[y:y+cols, x:x+rows])  # Check if any pixel is different
      if not is_overlapping:
        break
    
    # 背景上ROI區域摳出圖案mask
    mask = image_mask(mask)
    mask_inv = cv2.bitwise_not(mask)
    airport_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    crater_fg = cv2.bitwise_and(resized_image, resized_image, mask=mask)
    
    # 將「摳出圖案mask的背景」與「填充圖案的前景」相加
    dst = cv2.add(crater_fg,airport_bg)
    #cv2.imwrite("./dst.jpg", dst)

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

  mode = ['train_goo','test_goo'] #train/test
  bg_src = 'itri_small' #web/itri/itri_small
  
  dst_path = "./fusion_image"
  crater_dir = "./google_earth-1"
  bg_remove = "images_seg" #dirname prepocess crater saved
  
  for mod in modes:
    print(f"==========={mod}=============")
    crater_path = os.path.join(crater_dir, mod, bg_remove)
    mask_path = os.path.join(crater_dir, mod, 'mask')
    dstImg_path = os.path.join(dst_path,f"{mod}_{bg_src}",'images_whole')
    dstLab_path = os.path.join(dst_path,f"{mod}_{bg_src}",'labels_whole')
  
    if not os.path.exists(dstImg_path):
      os.makedirs(dstImg_path)
    if not os.path.exists(dstLab_path):
      os.makedirs(dstLab_path)
  
    if bg_src == 'itri_small':
      image_path = "image_airport.jpg"
    elif bg_src == 'itri':
      image_path = "runway-top-view_raw.tif"
    else:
      image_path = "Airport-plane-runway-top-view_3840x2160.jpg"
  
    if mod == 'train':
      fusion_image_number = 1
    else:
      fusion_image_number = 2
  
    # Fuse images
    for i in tqdm.tqdm(range(fusion_image_number)):
      # Random number of craters
      num_craters = random.randint(1, 10)  
      # Read background image
      image_airport = cv2.imread(image_path)
      
      if bg_src == 'itri':
        # Rotate the image by 45 degrees
        rotated_image = rotate_image(image_airport, -46.5)
        min_y, max_y = mask_road_region(rotated_image)
      
      airport_copy = image_airport.copy()                       
      label=[]
      
      # fusion back ground and crater
      dst, label_text = fusion_image(num_craters, image_airport, label)
      
      # saved images
      dst_image = os.path.join(dstImg_path,f"image_crater{i}.jpg")
      cv2.imwrite(dst_image, dst)
      # saved labels
      dst_label = os.path.join(dstLab_path,f"image_crater{i}.txt")
      with open(dst_label, 'w') as f:
        f.writelines(label_text)
      #print(f"saved_{i}")
  