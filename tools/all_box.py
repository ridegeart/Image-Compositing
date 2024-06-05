import cv2
import os
import csv

def convert_yolo_to_xyminmax(yolo_bbox):
    """Converts a YOLO bounding box format (x, y, w, h) to xmin, ymin, xmax, ymax format."""
    x, y, w, h = yolo_bbox
    width = 15785
    height = 1561
    xmin = int((x - w / 2)*width)
    ymin = int((y - h / 2)*height)
    xmax = int((x + w / 2)*width)
    ymax = int((y + h / 2)*height)
    return xmin, ymin, xmax, ymax

def compare_bboxes(gt_file, pred_file):
    """Compares ground truth (gt) and predicted (pred) bounding boxes and outputs undetected bboxes."""
    with open(gt_file, 'r') as f_gt:
        gt_bboxes = []
        for line in f_gt:
            label, x, y, w, h = line.split()
            gt_bbox = [float(x), float(y), float(w), float(h)]
            xmin, ymin, xmax, ymax = convert_yolo_to_xyminmax(gt_bbox)
            gt_bbox = [xmin, ymin, xmax, ymax]
            gt_bboxes.append(gt_bbox)

    with open(pred_file, 'r') as f_pred:
        pred_bboxes = []
        for line in f_pred:
            label, x, y, w, h = line.split()
            pred_bbox = [float(x), float(y), float(w), float(h)]
            xmin, ymin, xmax, ymax = convert_yolo_to_xyminmax(pred_bbox)
            pred_bbox = [xmin, ymin, xmax, ymax]
            pred_bboxes.append(pred_bbox)

    detected_bboxes = []
    if len(gt_bboxes) - len(pred_bboxes) != 0:
      for gt_bbox in gt_bboxes:
          found = False
          for pred_bbox in pred_bboxes:
              if iou(gt_bbox, pred_bbox) > 0.5:
                  found = True
                  xmin, ymin, xmax, ymax = gt_bbox
                  detected_bbox = [os.path.basename(gt_file), xmin, ymin, xmax, ymax]
                  detected_bboxes.append(detected_bbox)
                  break
    else:
      for gt_bbox in gt_bboxes:
        xmin, ymin, xmax, ymax = gt_bbox
        detected_bbox = [os.path.basename(gt_file), xmin, ymin, xmax, ymax]
        detected_bboxes.append(detected_bbox)
    return gt_bboxes

def iou(bbox1, bbox2):
    """Calculates the Intersection over Union (IoU) between two bounding boxes."""
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

    union = area1 + area2 - intersection
    iou = intersection / union
    return iou
    
def main():
    # Define ground truth and prediction folders
    gt_folder = "./datasets/fusion_image/train_itri_shadow/labels_whole"
    pred_folder = "./mmdetection/runs/exp16/labels"

    # Prepare output Excel file
    with open('all_box_itri.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'xmin', 'ymin', 'xmax', 'ymax'])
    
        # Iterate through ground truth and prediction files
        for gt_filename in os.listdir(gt_folder):
            if gt_filename.endswith('.txt'):
                gt_file = os.path.join(gt_folder, gt_filename)
                pred_filename = os.path.join(pred_folder, gt_filename)
                if os.path.isfile(pred_filename):
                    undetected_bboxes = compare_bboxes(gt_file, pred_filename)
    
                    # Write undetected bboxes to Excel file
                    for bbox in undetected_bboxes:
                        writer.writerow(bbox)
                    
if __name__ == "__main__":
    main()