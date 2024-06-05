import cv2
import os
import random
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from bg_preprocessing import mask_road_region,preprocessing,rotate_image,convert,image_mask

min_y = 0
max_y = 0

def load_image(image_path):
    crater_list = os.listdir(image_path)
    rnd_index = random.randint(0,len(crater_list)-1)
    f = crater_list[rnd_index]
    image_crater = cv2.imread(os.path.join(image_path,f), cv2.IMREAD_UNCHANGED)
    image_crater,[min_x,min_y,max_x,max_y] = preprocessing(image_crater)
    return image_crater[min_y:max_y+1,min_x:max_x+1]
  
# resize image by GSD
def resize_img(image,i):
    (h, w, d) = image.shape
    max_side = h if (h > w) else w
    min_size = 45 / GSD
    max_size = 1500 / GSD
    img_fx_min = min_size/max_side
    img_fx_max = max_size/max_side
    if int(i) == 1:
      size = img_fx_min
    else:
      size = random.triangular(img_fx_min,img_fx_max,img_fx_min) #random.uniform(img_fx_min, img_fx_max)
    return cv2.resize(image, dsize=None, fx=size, fy=size)

def fusion_image (i, image, label_text):

  if i <= 0:
    return image, label_text
    
  else:    
    # Read craters image
    image_crater = load_image(crater_path)
    
    # Loop until a non-overlapping location is found
    while True:
      # random crater size
      image_crater = resize_img(image_crater,i)
      cols, rows = image_crater.shape[:2]
      resized_image = rotate_image(image_crater, None)
      # Random location with upper bound check
      y = random.randint(1, image.shape[0] - cols)
      x = random.randint(1, image.shape[1] - rows-1)
      # Check for overlap with existing craters
      roi = image[y:y+cols, x:x+rows]
      is_overlapping = np.any(roi != airport_copy[y:y+cols, x:x+rows])
      if not is_overlapping:
        break
    
    # mask of bg remove crater
    crater_copy = resized_image.copy()
    mask = image_mask(crater_copy)

    # background ROI 區域摳出圖案mask
    mask_inv = cv2.bitwise_not(mask)
    airport_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    
    # 將「摳出圖案mask的背景」與「填充圖案的前景」相加
    dst = cv2.add(airport_bg, resized_image)
    
    #image = cv2.seamlessClone(resized_image, image, mask, (x+rows//2, y+cols//2), cv2.NORMAL_CLONE)

    # 用dst替換掉背景中含有彈坑的區域
    image[y:y+cols, x:x+rows] = dst
    
    # saving bounding box
    h= int(image.shape[0])
    w= int(image.shape[1])
    [x,y,w,h] = convert((w,h), (x, x+rows, y, y+cols))
    label = "0"+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h)+"\n"
    label_text.append(label)
    
    return fusion_image (i-1, image, label_text)

if __name__ == '__main__': 
  
  modes = ['train','test'] #train/test
  bg_src = 'itri' #web/itri/itri_shadow
  GSD = 1 #GSD of craters
  
  dst_path = "./fusion_image"
  crater_dir = "./Bomb-craters-low-1"
  bg_remove = "images_remove_bg" #dirname prepocess crater saved
  
  for mod in modes:
    print(f"==========={mod}=============")
    crater_path = os.path.join(crater_dir, mod, bg_remove)
    dstImg_path = os.path.join(dst_path,f"{mod}_{bg_src}",'images_whole')
    dstLab_path = os.path.join(dst_path,f"{mod}_{bg_src}",'labels_whole')
    
    if not os.path.exists(dstImg_path):
      os.makedirs(dstImg_path)
    if not os.path.exists(dstLab_path):
      os.makedirs(dstLab_path)
    
    if mod == 'train':
      fusion_fin_number = 1000
    else:
      fusion_fin_number = 200
    
    if bg_src == 'itri':
      image_path = "runway-top-view_raw.tif"
    elif bg_src == 'itri_shadow':
      image_path = "20240420_bg3.jpg"
    else:
      image_path = "Airport-plane-runway-top-view_3840x2160.jpg"
      
    for i in tqdm.tqdm(range(fusion_fin_number)):
      # Random number of craters
      num_craters = random.randint(1, 8)  
      # Read background image
      image_airport = cv2.imread(image_path)
      
      if bg_src == 'itri':
        rotated_image = rotate_image(image_airport, -46.5)
        min_y, max_y = mask_road_region(rotated_image)
        image_airport = rotated_image[min_y:max_y, 0:rotated_image.shape[1]]
      
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
  