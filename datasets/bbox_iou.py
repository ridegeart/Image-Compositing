import cv2
import os
import numpy as np

def getVOCNumbers(cur_big_width, cur_big_height, yolo_bbox):
    """Converts a YOLO bounding box format (x, y, w, h) to xmin, ymin, xmax, ymax format."""
    x, y, w, h = yolo_bbox
    width = cur_big_width
    height = cur_big_height
    xmin = int((x - w / 2)*width)
    ymin = int((y - h / 2)*height)
    xmax = int((x + w / 2)*width)
    ymax = int((y + h / 2)*height)
    return xmin, ymin, xmax, ymax

def iou(bbox1, bbox2):
    """Calculates the Intersection over Union (IoU) between two bounding boxes."""
    if len(bbox1) > 4:
      xmin1, ymin1, xmax1, ymax1, conf1 = bbox1
      xmin2, ymin2, xmax2, ymax2, conf2 = bbox2
    else:
      xmin1, ymin1, xmax1, ymax1 = bbox1
      xmin2, ymin2, xmax2, ymax2 = bbox2
    area1 = (xmax1-xmin1) * (ymax1-ymin1)
    area2 = (xmax2-xmin2) * (ymax2-ymin2)

    # Compute the intersection area
    inter_top = max(ymin1, ymin2)
    inter_left = max(xmin1, xmin2)
    inter_right = min(xmax1, xmax2)
    inter_bottom = min(ymax1, ymax2)
    
    if inter_right <= inter_left or inter_bottom <= inter_top:
      intersection = 0
    else:
        intersection = (inter_right-inter_left) * (inter_bottom-inter_top)

    if area1 == 0 or area2 == 0:
        iou = 0
    elif intersection == area1 or intersection == area2:
        iou = 1.0
    else:
        union = area1 + area2 - intersection
        iou = intersection / union
    return iou
    
def remove_overlapping_bboxes(bboxes, iou_threshold=0.5):
    """Removes highly overlapping bounding boxes from a list of bounding boxes.

    Args:
        bboxes (list of lists): A list of bounding boxes, each represented as a list of [x_min, y_min, x_max, y_max].
        iou_threshold (float): The minimum Intersection over Union (IoU) for considering two bounding boxes as overlapping.

    Returns:
        list of lists: The updated list of bounding boxes after removing overlaps.
    """
    # Iterate through all bounding boxes
    for i in range(len(bboxes)):
        for j in range(len(bboxes) - 1, i, -1):
            iou_score = iou(bboxes[i], bboxes[j])
            if iou_score >= iou_threshold:    
                xmin1, ymin1, xmax1, ymax1, conf1 = bboxes[i]
                xmin2, ymin2, xmax2, ymax2, conf2 = bboxes[j]
                
                xmin = min(xmin1, xmin2)
                ymin = min(ymin1, ymin2)
                xmax = max(xmax1, xmax2)
                ymax = max(ymax1, ymax2)
                conf = max(conf1, conf2) 
                
                bboxes[i] = [xmin, ymin, xmax, ymax, conf]
                bboxes.pop(j)
                #break
    return bboxes
    
def remove_overlap_boxes_txt(cur_big_width, cur_big_height, bboxes, iou_threshold=0.5):

    for i in range(len(bboxes)):
        for j in range(len(bboxes) - 1, i, -1):
            bboxes1 = getVOCNumbers(cur_big_width, cur_big_height, bboxes[i])
            bboxes2 = getVOCNumbers(cur_big_width, cur_big_height, bboxes[j])
            xmin1, ymin1, xmax1, ymax1 = bboxes1
            xmin2, ymin2, xmax2, ymax2 = bboxes2
            iou_score = iou(bboxes1, bboxes2)
            print(iou_score, i, j)
            if iou_score >= iou_threshold :

                xmin = min(xmin1, xmin2)
                ymin = min(ymin1, ymin2)
                xmax = max(xmax1, xmax2)
                ymax = max(ymax1, ymax2)
                
                w = xmax - xmin
                h = ymax - ymin
                x = xmin + w/2
                y = ymin + h/2
                
                bboxes[i] = [x/cur_big_width, y/cur_big_height, w/cur_big_width, h/cur_big_height]
                bboxes.pop(j)
                #break

    return bboxes