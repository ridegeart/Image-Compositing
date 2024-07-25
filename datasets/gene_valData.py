import os
import tqdm
import shutil

path = './fusion_image'
bg_src = 'itri_shadow' #web/itri/itri_shadow

bigImg_test = os.path.join(path,f"test_{bg_src}",'images_whole')
bigXml_test = os.path.join(path,f"test_{bg_src}",'label_xml_whole')
cropImg_test = os.path.join(path,f"test_{bg_src}",'images')
cropXml_test = os.path.join(path,f"test_{bg_src}",'label_xml')

bigImg_val = os.path.join(path,f"val_{bg_src}",'images_whole')
bigXml_val = os.path.join(path,f"val_{bg_src}",'label_xml_whole')
cropImg_val = os.path.join(path,f"val_{bg_src}",'images')
cropXml_val = os.path.join(path,f"val_{bg_src}",'label_xml')

def gene_file_list(bigImg_path, name_list_path):
    files = os.listdir(bigImg_path)
    for file in tqdm.tqdm(files):
        with open(name_list_path, 'a') as f:
            f.write(file[:-4] + '\n')

if not os.path.exists(bigImg_val):
        os.mkdir(bigImg_val)
if not os.path.exists(bigXml_val):
        os.mkdir(bigXml_val)
if not os.path.exists(cropImg_val):
        os.mkdir(cropImg_val)
if not os.path.exists(cropXml_val):
        os.mkdir(cropXml_val)

for i in tqdm.tqdm(range(100)):
    file_name = f"image_crater{i}.jpg"
    bigImg_path = os.path.join(bigImg_test,file_name)
    bigXml_path = os.path.join(bigXml_test,file_name[:-4]+'.xml')
    shutil.copy(bigImg_path,bigImg_val)
    shutil.copy(bigXml_path,bigXml_val)

    num_crop = 20

    for j in tqdm.tqdm(range(num_crop)):
        cropfile_name = f"image_crater{i}_{j}.jpg"
        cropImg_path = os.path.join(cropImg_test,cropfile_name)
        cropXml_path = os.path.join(cropXml_test,cropfile_name[:-4]+'.xml')
        shutil.copy(cropImg_path,cropImg_val)
        shutil.copy(cropXml_path,cropXml_val)

#gene file name list
print('Gene file name list ...')
name_list_path = os.path.join(path,f"val_{bg_src}","val.txt")
if not os.path.exists(name_list_path):
    gene_file_list(bigImg_val, name_list_path)
name_list_path = os.path.join(path,f"val_{bg_src}","val_cropped.txt")
if not os.path.exists(name_list_path):
    gene_file_list(cropImg_val, name_list_path)