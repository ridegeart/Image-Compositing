import cv2
import numpy as np
import random

def load_image_g(image_path, mask_path):
    crater_list = os.listdir(image_path)
    rnd_index = random.randint(0,len(crater_list)-1)
    f = crater_list[rnd_index]
    image_crater = cv2.imread(os.path.join(image_path,f))
    mask = cv2.imread(os.path.join(mask_path,f))
    return image_crater, mask

def rotate_image(image, angle):
    """Rotates an image by a given angle."""
    (h, w, d) = image.shape
    image_center = tuple(np.array(image.shape[:2]) / 2)
    if angle is None:
      angle = random.uniform(-40.0, 40.0) / 20.0
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image

def random_crop(image, min_y, max_y):
    """Randomly crops an image into a patch of the specified size."""
    patch_size=(3840, max_y - min_y)
    image_height, image_width = image.shape[:2]
    patch_x = random.randint(0, image_width - patch_size[0])
    patch_y = random.randint(min_y, max_y - patch_size[1])
    patch = image[patch_y:patch_y + patch_size[1], patch_x:patch_x + patch_size[0]]
    return patch
    
def preprocessing(image):
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

def mask_road_region(image):
    h = image.shape[0]
    w = image.shape[1]
    
    max_y = -float('INF')
    min_y = float('INF')
                
    for y in range(h):
        if image[y, 0, 0]>1:
            max_y = max(y,max_y)
            min_y = min(y,min_y)
    
    return min_y, max_y

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

def RandBackground(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    
    # Rotate the image by 45 degrees
    rotated_image = rotate_image(image, -46.5)
    
    min_y, max_y = mask_road_region(rotated_image)
    
    random_patch = random_crop(rotated_image, min_y, max_y)
    random_patch = cv2.resize(random_patch, (3840,2160))
    return random_patch
    
if __name__ == "__main__":
    # Input image path
    image_path = "./fusion_image/test/20240420.tif"

    # Read the input image
    image = cv2.imread(image_path)

    # Rotate the image by 45 degrees
    rotated_image = rotate_image(image, -47)
    
    min_y, max_y = mask_road_region(rotated_image)
    
    image_bg = rotated_image[min_y:max_y, 0:rotated_image.shape[1]]
    
    cv2.imwrite("./fusion_image/test/20240420_remove.jpg", image_bg)
    
    cv2.imwrite("./fusion_image/test/20240420_bg.jpg", image_bg[4206:5767, 0:rotated_image.shape[1]])
    
    '''
    # Generate and save random crops
    for i in range(10):  # Generate 10 random crops
        random_patch = random_crop(rotated_image, min_y, max_y)
        random_patch = cv2.resize(random_patch, (3840,2160))
        cv2.imwrite(f"patch_{i}.jpg", random_patch)
    '''
