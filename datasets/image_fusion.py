import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image,ImageDraw,ImageFont

# Read airport image
image_airport = cv2.imread('Airport-plane-runway-top-view_3840x2160.jpg')
    
# 比例縮小圖檔
def resize_img(image):
    size = random.uniform(0.2, 1.0)
    return cv2.resize(image, dsize=None, fx=size, fy=size)

# 旋轉圖片
def rotate_img(image):
    (h, w, d) = image.shape
    center = (w // 2, h // 2)
    angle = random.randint(-40, 40) / 20
    M = cv2.getRotationMatrix2D(center, angle, 1.04)
    image = cv2.warpAffine(image, M, (w, h))
    return image

def fusion_image (i, image):

  if i <= 0:
    return image
    
  else:    
    # 讀取彈坑圖片
    image_crater = cv2.imread(f"warped_image_opt.png")
    max_cols, max_rows = image_crater.shape[:2]
    
    # random crater size
    #resized_image = resize_img(image_crater, rows, cols)
    resized_image = resize_img(image_crater)
    cols, rows = resized_image.shape[:2]
    resized_image = rotate_img(resized_image)
    
    while True:
      # Random location with upper bound check
      y = random.randint(0, image.shape[0] - cols-1)
      x = random.randint(0, image.shape[1] - rows-1)
  
      # Check for overlap with existing craters (assuming craters are opaque)
      roi = image[y:y+cols, x:x+rows] #1
      is_overlapping = np.any(roi != image_airport[y:y+cols, x:x+rows])  # Check if any pixel is different
  
      if not is_overlapping:
        break
    
    im_mask = np.full(roi.shape, 255, dtype = np.uint8) #2
    
    #image = cv2.seamlessClone(resized_image, image, im_mask, (x+rows//2, y+cols//2), cv2.NORMAL_CLONE)
    #image = cv2.seamlessClone(resized_image, image, im_mask, (x+rows//2, y+cols//2), cv2.NORMAL_CLONE) #1
    image[y:y+cols, x:x+rows] = cv2.addWeighted(image[y:y+cols, x:x+rows], 0.2, resized_image, 0.8, 0) #2
    #image[y:y+cols, x:x+rows] = cv2.addWeighted(blurred_image, 0.8, mask, 0.01, 0)
    '''
    roi[:] = cv2.add(crater_fg, airport_bg)
    image[y:y+cols, x:x+rows] = roi[:]
    '''
    return fusion_image (i-1, image)
    
if __name__ == '__main__': 
  # Random number of craters
  num_craters = random.randint(1, 5)  
  
  # Fuse images
  dst = fusion_image(num_craters, image_airport)
  
  # saved images
  os.chdir('./fusion_craters')
  cv2.imwrite("image10.jpg", dst)
  print("saved")