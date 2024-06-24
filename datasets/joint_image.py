import os
import cv2
from bbox_iou import remove_overlap_boxes_txt

# todo: img_info={'name': [w, h]}
img_info = {}
all_info = {}


# initial img_info
def init(big_images_path):
    image_names = os.listdir(big_images_path)
    for img_name in image_names:
        big_path = big_images_path + '/' + img_name
        img = cv2.imread(big_path)
        size = img.shape[0:2]
        w = size[1]
        h = size[0]
    
        img_info[img_name.split('.')[0]] = [w, h]

# Integrate all detect croped labels into original image
def get_label_info(labels_path, cur_img_width, cur_img_height):
    # read all croped label
    labels = os.listdir(labels_path)
    for label in labels:
        cur_label_belong = label.split('_')[0] + '_' + label.split('_')[1]
        # label info form
        child_class_index = []
        child_x = []
        child_y = []
        child_width = []
        child_height = []
        
        cur_big_height = img_info[cur_label_belong][1]
        cur_big_width = img_info[cur_label_belong][0]

        # read one croped label file
        f = open(labels_path + '/' + label, 'r')
        lines = f.read()
        f.close()
        contents = lines.split('\n')[:-1]
        
        for content in contents:
            content = content.split(' ')
            # read yolo form
            class_index = int(content[0])
            x = float(content[1])
            y = float(content[2])
            width = float(content[3])
            height = float(content[4])
            pass
            assert class_index != -1 or x != -1.0 or y != -1.0 or width != -1.0 or height != -1.0, \
                f'class_index:{class_index}, x:{x}, y:{y}, width:{width}, height:{height}'
            
            num = label.split('_')[-1].split('.')[0]  # xxxx_x.jpg  xxxx_mix_row_xx.jpg xxxx_mix_col_xx.jpg
            distance_x = 0
            distance_y = 0

            step = int(cur_big_width // (cur_img_width/2))
            stepy = int(cur_big_height // (cur_img_height/2))
            
            # according to [num] mormalized the label
            distance_x = int(num) % step
            distance_y = int(num) // step
            
            y = y * cur_img_height + distance_y * cur_img_height / 2 if distance_y != (stepy - 1) else y * cur_img_height + (cur_big_height - cur_img_height)
            x = x * cur_img_width + distance_x * cur_img_width / 2 if distance_x != (step - 1) else x * cur_img_width + (cur_big_width - cur_img_width)
            width = width * cur_img_width
            height = height * cur_img_height
            
            assert x < icur_big_width and y < cur_big_height, f'{label}, {content}\n w:{cur_big_width}, h:{cur_big_height}, x:{x}, y:{y}'
            assert x != 0.0 or y != 0.0 or width != 0.0 or height != 0.0, f'x:{x}, y:{y}, width:{width}, height:{height}'
            child_class_index.append(class_index)
            child_x.append(x)
            child_y.append(y)
            child_width.append(width)
            child_height.append(height)
        # todo: according to cur_label_belong save in all_info
        for index, x, y, width, height in zip(child_class_index, child_x, child_y, child_width, child_height):
            if cur_label_belong not in all_info:
                all_info[cur_label_belong] = [[index, x, y, width, height]]
            else:
                all_info[cur_label_belong].append([index, x, y, width, height])
        child_class_index.clear()
        child_x.clear()
        child_y.clear()
        child_width.clear()
        child_height.clear()

# todo: transform to yolo
def save_yolo_label(yolo_labels_path):
    for key in all_info:
        yolo_label_path = yolo_labels_path + key + '.txt'
        cur_big_height = img_info[key][1]
        cur_big_width = img_info[key][0]
        bboxes = []
        for index, x, y, width, height in all_info[key]:
            x = x / cur_big_width
            y = y / cur_big_height
            width = width / cur_big_width
            height = height / cur_big_height
            bboxes.append([x, y, width, height])
            assert x < 1.0 and y < 1.0 and width < 1.0 and height <= 1.0, f'{key} {index}\nx:{x}, y:{y}, width:{width}, height:{height}'

        non_overlapping_bboxes = remove_overlap_boxes_txt(cur_big_width, cur_big_height, bboxes)
        with open(yolo_label_path, 'w') as f:
            for bbox in non_overlapping_bboxes:
                f.write('0 '+ ' '.join([str(val) for val in bbox]) + '\n')


def joint_main(big_images_path='./fusion_image/val/images_bg',
               labels_path='/home/training/mmdetection/runs/exp27/labels',
               yolo_labels_path='/home/training/mmdetection/runs/exp27/labels_croped_fusion/',
               cur_img_width = 3000,
               cur_img_height = 1220):
    print(f'融合图片, 原图片路径：{big_images_path}\n小图检测的txt结果路径：{labels_path}\n数据融合后txt结果路径：{yolo_labels_path}')
    init(big_images_path)
    get_label_info(labels_path, cur_img_width, cur_img_height)
    save_yolo_label(yolo_labels_path)

joint_main()
