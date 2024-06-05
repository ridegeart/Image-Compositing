import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
import subprocess

def load_data(ground_truth_xml):
  
  ground_truth = []
  bbox = []
  file = os.path.basename(ground_truth_xml)
  tree = ET.parse(ground_truth_xml)
  root = tree.getroot()
  
  for obj in root.findall('object'):
    bnd_box = obj.find('bndbox')
    polygon = obj.find('polygon')

    x = {}
    y = {}
    
    x['x1'] = int(float(polygon.find('x1').text))
    x['x2'] = int(float(polygon.find('x2').text))
    x['x3'] = int(float(polygon.find('x3').text))
    x['x4'] = int(float(polygon.find('x4').text))
    
    y['y1'] = int(float(polygon.find('y1').text))
    y['y2'] = int(float(polygon.find('y2').text))
    y['y3'] = int(float(polygon.find('y3').text))
    y['y4'] = int(float(polygon.find('y4').text))
    
    xmin = int(bnd_box.find('xmin').text)
    ymin = int(bnd_box.find('ymin').text)
    xmax = int(bnd_box.find('xmax').text)
    ymax = int(bnd_box.find('ymax').text)
    
    dif_y = (ymax - ymin)/2
    dif_x = (xmax - xmin)/2
    
    p1, p2, p3, p4 = (x['x1'],y['y1']), (x['x2'],y['y2']), (x['x3'],y['y3']), (x['x4'],y['y4'])

    for i, point_y in enumerate(list(y.values())):
      if point_y - ymin +2 < dif_y :
        if list(x.values())[i] - xmin +2 < dif_x:
          p1 = (list(x.values())[i],point_y)
        else:
          p2 = (list(x.values())[i],point_y)
      else:
        if list(x.values())[i] - xmin +2 < dif_x:
          p4 = (list(x.values())[i],point_y)
        else:
          p3 = (list(x.values())[i],point_y)
    
    ground_truth.append({
      'file': file,
      'p1': p1,
      'p2': p2,
      'p3': p3,
      'p4': p4,
      })

  return ground_truth
# perspective transform to top view
def perspective_transform(image, tl, tr, br, bl):
  """
  perspective_transform

  Args:
    image: image to transform
    src_points(tl, tr, br, bl): for source point read from label file ,Format [x, y]

  Returns:
    image transformed
  """
  height, width = image.shape[:2]
  
  # Here, I have used L2 norm. You can use L1 also.
  width_AD = np.sqrt(((tl[0] - tr[0]) ** 2) + ((tl[1] - tr[1]) ** 2))
  width_BC = np.sqrt(((bl[0] - br[0]) ** 2) + ((bl[1] - br[1]) ** 2))
  maxWidth = max(int(width_AD), int(width_BC))
   
   
  height_AB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  height_CD = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))
  maxHeight = max(int(height_AB), int(height_CD))
  
  #apply geometrical transformation
  pts1 = np.float32([tl, tr, br, bl])
  pts2 = np.float32([(0, 0), (width, 0), (width, height), (0, height)])
  #apply geometrical transformation
  output_pts = np.float32([[0, 0],
                          [maxWidth - 1, 0],
                          [maxWidth - 1, maxHeight - 1],
                          [0, maxHeight - 1]])
                          
  matrix = cv2.getPerspectiveTransform(pts1, output_pts)
  warped_image = cv2.warpPerspective(image, matrix, (maxWidth - 1, maxHeight - 1))
  warped_image = cv2.resize(warped_image, dsize=(height,height), interpolation=cv2.INTER_NEAREST)

  return warped_image
# backfround remove
def bg_remove(crater_path, remove_bg_path):

  for f in os.listdir(crater_path):
    crater = os.path.join(crater_path,f)
    output = remove_bg_path+f[:-4]+".png"
    subprocess.run(['backgroundremover', '-m' , 'u2net', '-i', crater, '-o', output])
  
if __name__ == '__main__': 

  src_path = "./Bomb-craters-low-1/test/images"  
  gnd_path = "./Bomb-craters-low-1/test/labels/" #endwith '/'
  dst_path = "./Bomb-craters-low-1/test/images_seg1"
  remove_bg_path = "./Bomb-craters-low-1/test/images_remove_bg_1/" #endwith '/'
  
  if not os.path.exists(dst_path):
    os.mkdir(dst_path)
  
  if os.path.isfile(src_path):
    f = os.path.basename(src_path)
    image = cv2.imread(os.path.join(src_path))
    gnd_truth = load_data(gnd_path+f[:-4]+'.xml')
    # perspective_transform
    warped_image = perspective_transform(image, gnd_truth[0]['p1'], gnd_truth[0]['p2'], gnd_truth[0]['p3'], gnd_truth[0]['p4'])
    cv2.imwrite(os.path.join(dst_path,f), warped_image)
    
  elif os.path.isdir(src_path):
  
    for f in os.listdir(src_path):
      image = cv2.imread(os.path.join(src_path,f))
      gnd_truth = load_data(gnd_path+f[:-4]+'.xml')
      print(f)
      # perspective_transform
      warped_image = perspective_transform(image, gnd_truth[0]['p1'], gnd_truth[0]['p2'], gnd_truth[0]['p3'], gnd_truth[0]['p4'])
      cv2.imwrite(os.path.join(dst_path,f), warped_image)
  else:
      print("it's a special file(socket,FIFO,device file)")
  
  if not os.path.exists(remove_bg_path):
    os.mkdir(remove_bg_path)
  # backfround remove
  bg_remove(dst_path, remove_bg_path)
  

