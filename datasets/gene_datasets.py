import os
import tqdm
from txt2xml import convert_txt_to_xml
from image_cropped_train import crop_dataset

def gene_file_list(bigImg_path, name_list_path):
    files = os.listdir(bigImg_path)
    for file in tqdm.tqdm(files):
        with open(name_list_path, 'a') as f:
            f.write(file[:-4] + '\n')

def yolo2Voc(bigLab_path, bigImg_path, xml_dir):
    for file in tqdm.tqdm(os.listdir(bigLab_path)):
        txt_file = os.path.join(bigLab_path, file)
        xml_file = os.path.join(xml_dir, file[:-4] + '.xml')
        img_file = os.path.join(bigImg_path, file[:-4] + '.jpg')
        convert_txt_to_xml(txt_file, xml_file, img_file)

if __name__ == '__main__': 
    path = './fusion_image'
    modes = ['train','test']
    bg_src = 'itri_shadow' #web/itri/itri_shadow
    sliding_window = False
    scale = 1
    
    for mod in modes:
      print(f"==========={mod}=============")
      bigImg_path = os.path.join(path,f"{mod}_{bg_src}",'images_whole')
      bigLab_path = os.path.join(path,f"{mod}_{bg_src}",'labels_whole')
      
      #gene file name list
      name_list_path = os.path.join(path,f"{mod}_{bg_src}",'{}.txt'.format(mod))
      gene_file_list(bigImg_path, name_list_path)
      
      #Yolo to VOC
      xml_dir = os.path.join(path,f"{mod}_{bg_src}",'label_xml_whole')
      if not os.path.exists(xml_dir):
        os.mkdir(xml_dir)
      yolo2Voc(bigLab_path, bigImg_path, xml_dir)
      
      #Sliding window
      if sliding_window:
        
        if bg_src == 'itri' or bg_src == 'itri_shadow':
          windows_width = 3840
        else:
          windows_width = 256
        
        cropImg_path = os.path.join(path,f"{mod}_{bg_src}",'images')
        cropLab_path = os.path.join(path,f"{mod}_{bg_src}",'labels')
        if not os.path.exists(cropImg_path):
          os.makedirs(cropImg_path)
        if not os.path.exists(cropLab_path):
          os.makedirs(cropLab_path)
        
        crop_dataset(bigImg_path, scale, windows_width, bigLab_path,  name_list_path, cropLab_path, cropImg_path, sub='')
        
        #gene windows file name list
        name_list_path = os.path.join(path,f"{mod}_{bg_src}",'{}_cropped.txt'.format(mod))
        gene_file_list(cropImg_path, name_list_path)
        
        #Yolo to VOC
        xml_dir = os.path.join(path,f"{mod}_{bg_src}",'label_xml')
        if not os.path.exists(xml_dir):
          os.mkdir(xml_dir)
        yolo2Voc(cropLab_path, cropImg_path, xml_dir)