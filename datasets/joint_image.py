import os
import cv2
from datasets.bbox_iou import remove_overlap_boxes_txt, getVOCNumbers

# todo: img_info={'name': [w, h]}
img_info = {}
all_info = {}


# initial img_info
def init(big_images_path):
    for image_name in os.listdir(big_images_path):
        img = cv2.imread(os.path.join(big_images_path, image_name))
        size = img.shape[0:2]
        w = size[1]
        h = size[0]
        
        img_info[image_name.split('.')[0]] = [w, h]

# Integrate all detect croped labels into original image
def get_label_info(labels_path, scale, cur_img_width, cur_img_height):
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
        
        num = -1
        class_index = -1
        x = 0.0
        y = 0.0
        width = 0.0
        height = 0.0
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
            #cur_img_width = 256 #img_info[cur_label_belong][0]
            #cur_img_height = img_info[cur_label_belong][1]
            step = int(cur_big_width // (cur_img_width/2))
            stepy = int(cur_big_height // (cur_img_height/2))
            
            # according to [num] mormalized the label
            distance_x = int(num) % step
            distance_y = int(num) // step
            
            if distance_y == (stepy-1):
                y = y * cur_img_height + (img_info[cur_label_belong][1] - cur_img_height)
            else:
                y = y * cur_img_height + distance_y * cur_img_height / 2
            if distance_x == (step-1):
                x = x * cur_img_width + (img_info[cur_label_belong][0] - cur_img_width)
            else:
                x = x * cur_img_width + distance_x * cur_img_width / 2
            
            assert cur_img_width != 0 or cur_img_height != 0 or distance_x != 0 or distance_y != 0, \
                f'cur_img_width:{cur_img_width}, cur_img_height:{cur_img_height}, distance_x:{distance_x}, distance_y:{distance_y}'
            assert x < img_info[cur_label_belong][0] and y < img_info[cur_label_belong][1], f'{label}, {content}\n w:{img_info[cur_label_belong][0]}, h:{img_info[cur_label_belong][1]}, x:{x}, y:{y}'
            width = width * cur_img_width
            height = height * cur_img_height
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

def output_result(bboxes):
    for key in all_info:
        cur_big_height = img_info[key][1]
        cur_big_width = img_info[key][0]
        bboxes_VOC = []

        for bbox in bboxes:
            xmin, ymin, xmax, ymax = getVOCNumbers(cur_big_width, cur_big_height, bbox)
            p1 = [xmin,ymin]
            p2 = [xmin,ymax]
            p3 = [xmax,ymax]
            p4 = [xmax,ymin]
            bboxes_VOC.append([p1, p2, p3, p4])

        # Create the JSON structure
        json_data = {
            "result": {
                "potholes": [
                    {
                        "bbox": bboxes_VOC
                    }
                ]
            },
            "status": {
                "code": 1,
                "message": "Get image successfully."
            },
            "version": "XXXXXX"
        }

    return json_data

# todo: transform to yolo
def save_yolo_label(yolo_labels_path):
    if os.path.exists(yolo_labels_path) is False:
        os.mkdir(yolo_labels_path)
    for key in all_info:
        yolo_label_path = os.path.join(yolo_labels_path,'{}.txt'.format(key))
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
            #content += f'{index} {x} {y} {width} {height}\n'
        #print(key, len(bboxes))
        non_overlapping_bboxes = remove_overlap_boxes_txt(cur_big_width, cur_big_height, bboxes)
        with open(yolo_label_path, 'w') as f:
            for bbox in non_overlapping_bboxes:
                f.write('0 '+ ' '.join([str(val) for val in bbox]) + '\n')
                #f.write(content)

        result = output_result(non_overlapping_bboxes)
        return result

def joint_main(big_images_path, labels_path, yolo_labels_path, cur_img_width, cur_img_height,
               scale=1):
    print(f'融合图片, 原图片路径：{big_images_path}\n小图检测的txt结果路径：{labels_path}\n数据融合后txt结果路径：{yolo_labels_path}')
    init(big_images_path)
    get_label_info(labels_path, scale, cur_img_width, cur_img_height)
    predict_result = save_yolo_label(yolo_labels_path)
    return predict_result

if __name__ == '__main__':
    big_images_path='./fusion_image/val/images_bg',
    labels_path='/home/training/mmdetection/runs/exp27/labels',
    yolo_labels_path='/home/training/mmdetection/runs/exp27/labels_croped_fusion/',
    cur_img_width = 3000,
    cur_img_height = 1220,
    joint_main(big_images_path, labels_path, yolo_labels_path, cur_img_width, cur_img_height,
                scale=1)
