import os
import argparse
import numpy as np
from terminaltables import AsciiTable
from mmcv.utils import print_log
from mean_ap import average_precision

def load_data(ground_truth_dir, prediction_dir):
 """
 加載 ground truth 和 prediction 資料

 Args:
  ground_truth_dir: 地面真相資料夾
  prediction_dir: 預測結果資料夾

 Returns:
  ground_truth, predictions
 """
 width = 20970
 height = 1561
 ground_truth = []
 for file in os.listdir(ground_truth_dir):
  with open(os.path.join(ground_truth_dir, file)) as f:
   for line in f.readlines():
    label, x, y, w, h = line.split()
    ground_truth.append({
     'file': file,
     'label': int(label),
     'top': (float(y)-float(h)/2)*height,
     'left': (float(x)-float(w)/2)*width,
     'bottom': (float(y)+float(h)/2)*height,
     'right': (float(x)+float(w)/2)*width,
    })

 predictions = []
 for file in os.listdir(prediction_dir):
  with open(os.path.join(prediction_dir, file)) as f:
   for line in f.readlines():
    label, x, y, w, h = line.split()
    predictions.append({
     'file': file,
     'label': int(label),
     'top': (float(y)-float(h)/2)*height,
     'left': (float(x)-float(w)/2)*width,
     'bottom': (float(y)+float(h)/2)*height,
     'right': (float(x)+float(w)/2)*width,
    })

 return ground_truth, predictions

def calculate_iou(box1, box2):
 inter_top = max(box1['top'], box2['top'])
 inter_left = max(box1['left'], box2['left'])
 inter_bottom = min(box1['bottom'], box2['bottom'])
 inter_right = min(box1['right'], box2['right'])
 
 area1 = (box1['right'] - box1['left']) * (box1['bottom'] - box1['top'])
 area2 = (box2['right'] - box2['left']) * (box2['bottom'] - box2['top'])
  
 if inter_right <= inter_left or inter_bottom <= inter_top:
  return 0
 else:
  intersection = (inter_right-inter_left) * (inter_bottom-inter_top)
  
 if area1 == 0 or area2 == 0:
  iou = 0
 else:
  union = area1 + area2 - intersection
  iou = intersection / union
 return iou

def calculate_recall_iou(ground_truth, predictions, iou_threshold = 0.5):
 """
 計算 RecallIOU

 Args:
  ground_truth: 地面真相 (list of bounding boxes)
  predictions: 預測結果 (list of bounding boxes)

 Returns:
  RecallIOU
 """
 class_metrics = {}
 matched_count = 0
 total_ground_truth = 0
 for ground_truth_box in ground_truth:
  # 找到每個 ground truth 框匹配的 best prediction 框
  best_iou = -float('inf')
  best_prediction_box = None
  for prediction_box_item in predictions:
   if prediction_box_item['file'] == ground_truth_box['file']:
    iou = calculate_iou(ground_truth_box, prediction_box_item)
    if iou > best_iou and iou >= iou_threshold: # 閾值設為 0.5
     best_iou = iou
     best_prediction_box = prediction_box_item

  total_ground_truth += 1
  if best_prediction_box is not None and ground_truth_box['label'] == best_prediction_box['label']:
   matched_count += 1

  label = ground_truth_box['label']
  class_metrics.setdefault(label, {'matched_count': 0, 'total_ground_truth': 0})
  class_metrics[label]['total_ground_truth'] += 1
  if best_prediction_box is not None and ground_truth_box['label'] == best_prediction_box['label']:
   class_metrics[label]['matched_count'] += 1

 return matched_count / total_ground_truth, class_metrics

def calculate_precision_iou(ground_truth, predictions, iou_threshold = 0.5):
 """
 計算 PrecisionIOU

 Args:
  ground_truth: 地面真相 (list of bounding boxes)
  predictions: 預測結果 (list of bounding boxes)

 Returns:
  PrecisionIOU
 """
 class_metrics = {}
 matched_count = 0
 total_predictions = 0
 for prediction_box in predictions:
  # 找到對應的 ground truth 框
  best_iou = -float('inf')
  bset_gt_box = None
  for ground_truth_box_item in ground_truth:
   if prediction_box['file'] == ground_truth_box_item['file']:
    iou = calculate_iou(ground_truth_box_item, prediction_box)
    if iou > best_iou and iou >= iou_threshold: # 閾值設為 0.5
     best_iou = iou
     bset_gt_box = ground_truth_box_item

  total_predictions += 1
  if bset_gt_box is not None and prediction_box['label'] == bset_gt_box['label']:
   matched_count += 1
   
  label = prediction_box['label']
  class_metrics.setdefault(label, {'matched_count': 0, 'total_predictions': 0})
  class_metrics[label]['total_predictions'] += 1
  if bset_gt_box is not None and prediction_box['label'] == bset_gt_box['label']:
   class_metrics[label]['matched_count'] += 1
  
 return matched_count / total_predictions, class_metrics


def evaluate(ground_truth, predictions):
 """
 評估模型的性能

 Args:
  ground_truth: 地面真相 (list of labels)
  predictions: 預測結果 (list of labels)

 Returns:
  RecallIOU, PrecisionIOU, SCOREdis
 """

 recall_iou, class_recall_metrics = calculate_recall_iou(ground_truth, predictions)
 precision_iou, class_precision_metrics = calculate_precision_iou(ground_truth, predictions)
 f1_scores = calculate_f1_score(recall_iou, precision_iou)
  
 class_f1_scores = {}
 for label, metrics in class_recall_metrics.items():
  recall = metrics['matched_count'] / metrics['total_ground_truth']
  precision = class_precision_metrics[label]['matched_count'] / class_precision_metrics[label]['total_predictions']
  f1_score = 2 * (recall * precision) / (recall + precision)
  class_f1_scores[label] = f1_score

 return recall_iou, precision_iou, f1_scores, class_f1_scores, class_recall_metrics, class_precision_metrics

def calculate_f1_score(recall_iou, precision_iou):
 """
 計算 F1 score

 Args:
  recall_iou: RecallIOU 值
  precision_iou: PrecisionIOU 值

 Returns:
  F1 score
 """

 # 防止分母為 0
 if precision_iou + recall_iou == 0:
  return 0

 return 2 * (precision_iou * recall_iou) / (precision_iou + recall_iou)

def calculate_map(ground_truth, predictions, iou_thresholds=[0.5]):
 """
 Calculates mean Average Precision (mAP) for object detection.

 Args:
     ground_truth (list): List of ground truth bounding boxes.
     predictions (list): List of predicted bounding boxes.
     iou_thresholds (list): List of IoU thresholds for calculating AP at different levels.

 Returns:
     float: The overall mAP value.
 """
 class_aps = {}
 for class_label in {ground_truth_box['label'] for ground_truth_box in ground_truth}:
  recalls = []
  precisions = []
  for iou_threshold in iou_thresholds:
   # ... (calculate recalls and precisions for each IoU threshold)
   recall, _ = calculate_recall_iou(
          [gt for gt in ground_truth if gt['label'] == class_label], predictions, iou_threshold)
   precision, _ = calculate_precision_iou(
          [gt for gt in ground_truth if gt['label'] == class_label], predictions, iou_threshold)
   recalls.append(recall)
   precisions.append(precision)
   
 recalls_array = np.array(recalls)
 precisions_array = np.array(precisions)

 class_aps[class_label] = average_precision(recalls_array, precisions_array, mode='area')

 mAP = sum(class_aps.values()) / len(class_aps.values())
 return mAP

if __name__ == '__main__':
 # 載入 ground truth 和 prediction 資料
  
 prediction_dir = "./mmdetection/runs/exp22/labels_croped_fusion/"
 ground_truth_dir = "./datasets/fusion_image/test/labels_itri_bg"

 ground_truth, predictions = load_data(ground_truth_dir, prediction_dir)

 recall_iou, precision_iou, f1_scores, class_f1_scores, class_recall, class_precision = evaluate(ground_truth, predictions)
 mean_ap = calculate_map(ground_truth, predictions)
 print('mAP:', mean_ap)
 print('RecallIOU:', recall_iou)
 print('PrecisionIOU:', precision_iou)
 print('F1 scores:', f1_scores)
 header = ['class', 'gts', 'dets', 'recall', 'precision', 'f1 scores']  #打印precision??
 table_data = [header]
 for label, f1_score in class_f1_scores.items():
  row_data = [ 'crater', class_recall[label]['total_ground_truth'], class_precision[label]['total_predictions'],
                  f'{recall_iou:.3f}', f'{precision_iou:.3f}', f'{f1_scores:.3f}'  #打印precision的值
                  ]
  table_data.append(row_data)
 table_data.append(['mAP', '', '', '', '', f'{mean_ap:.3f}'])
 table = AsciiTable(table_data)
 table.inner_footing_row_border = True
 print_log('\n' + table.table)
 #print(f'{label}: {f1_score}')