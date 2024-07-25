import os
import cv2
import pickle
import numpy as np
from copy import deepcopy
from bbox_iou import remove_overlapping_bboxes

# todo: img_info={'name': [w, h]}
img_info = {}
all_info = {}

# initial img_info
def init(big_images_path):
    image_names = os.listdir(big_images_path)
    for img_name in image_names:
        big_path = os.path.join(big_images_path, img_name)
        img = cv2.imread(big_path)
        size = img.shape[0:2]
        w = size[1]
        h = size[0]
        img_info[img_name.split('.')[0]] = [w, h]

# Integrate all detect croped labels into original image
def get_label_info(result_path, name_file_list, cur_img_width, cur_img_height):
    # read all croped name_file_list
    f = open(name_file_list, 'r')
    lines = f.read()
    f.close()
    names = lines.split('\n')
    result = pickle.load(open(result_path,'rb'))
    for i in range(len(result)):
        cur_label_belong = names[i].split('_')[0] + '_' + names[i].split('_')[1]
        
        # label info form
        child_confidence = []
        child_xmin = []
        child_ymin = []
        child_xmax = []
        child_ymax = []
        
        num_class = len(result[i])
        cur_big_height = img_info[cur_label_belong][1]
        cur_big_width = img_info[cur_label_belong][0]
        
        for class_id in range(num_class):
            for box in result[i][class_id]:
                # read yolo form
                confidence = float(box[4])
                xmin = float(box[0])
                ymin = float(box[1])
                xmax = float(box[2])
                ymax = float(box[3])
                pass
                assert confidence != -1 or xmin != -1.0 or ymin != -1.0 or xmax != -1.0 or ymax != -1.0, \
                    f'class_index:{class_id}, xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}'
                
                num = names[i].split('_')[-1]
                distance_x = 0
                distance_y = 0

                step = int(cur_big_width // (cur_img_width/2))
                stepy = int(cur_big_height // (cur_img_height/2))
                
                # according to [num] mormalized the label
                distance_x = int(num) % step
                distance_y = int(num) // step
                
                xmin = xmin + distance_x * cur_img_width / 2 if distance_x != (step - 1) else xmin + (cur_big_width - cur_img_width)
                ymin = ymin + distance_y * cur_img_height / 2 if distance_y != (stepy - 1) else ymin + (cur_big_height - cur_img_height)
                xmax = xmax + distance_x * cur_img_width / 2 if distance_x != (step - 1) else xmax + (cur_big_width - cur_img_width)
                ymax = ymax + distance_y * cur_img_height / 2 if distance_y != (stepy - 1) else ymax + (cur_big_height - cur_img_height)

                assert xmax <= cur_big_width and ymax <= cur_big_height, f'{num}, {box}\n w:{cur_big_width}, h:{cur_big_height}, xmax:{xmax}, ymax:{ymax}'
                assert xmin != 0.0 or ymin != 0.0 or xmax != 0.0 or ymax != 0.0, f'xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}'
                child_confidence.append(confidence)
                child_xmin.append(xmin)
                child_ymin.append(ymin)
                child_xmax.append(xmax)
                child_ymax.append(ymax)

            if cur_label_belong not in all_info:
                all_info[cur_label_belong] = []

            # todo: according to cur_label_belong save in all_info
            for confi, xmin, ymin, xmax, ymax in zip(child_confidence, child_xmin, child_ymin, child_xmax, child_ymax):  
                all_info[cur_label_belong].append([xmin, ymin, xmax, ymax, confi])
                
            child_confidence.clear()
            child_xmin.clear()
            child_ymin.clear()
            child_xmax.clear()
            child_ymax.clear()
        
# todo: saved in pickle
def save_new_pickle(joint_name_list, joint_path):
    f = open(joint_name_list, 'r')
    lines = f.read()
    f.close()
    names = lines.split('\n')[:-1]
    combined_results = []
   
    for name in names:
        print(name)
        bboxes = all_info[name]
        new_bboxes = remove_overlapping_bboxes(bboxes)
        new_bboxes = np.array(new_bboxes, dtype='float32')
        if new_bboxes.size == 0:
            new_bboxes =  np.empty((0, 5), dtype='float32')
        image_results = [new_bboxes]
        combined_results.append(image_results.copy())

    with open(joint_path, 'wb') as f:
        pickle.dump(combined_results, f)

def joint_main(big_images_path='./fusion_image/test_itri_shadow1/images_whole',
               cropped_name_list='./fusion_image/test_itri_shadow1/test_cropped.txt',
               joint_name_list='./fusion_image/test_itri_shadow1/test.txt',
               result_path='/home/training/mmdetection/results_mmdet/results_test_shadow1_exp14.pkl',
               joint_path='/home/training/mmdetection/results_mmdet/results_test_shadow1_exp14_joint.pkl',
               cur_img_width = 3000,
               cur_img_height = 1220,):
    print(f'融合圖片，元圖片路徑：{big_images_path}\n小圖的name file list：{cropped_name_list}\n檢測的pickle結果檔：{result_path}\n元圖的name file list：{joint_name_list}')
    init(big_images_path)
    get_label_info(result_path, cropped_name_list, cur_img_width, cur_img_height)
    save_new_pickle(joint_name_list, joint_path)

joint_main()