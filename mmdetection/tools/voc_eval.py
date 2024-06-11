from argparse import ArgumentParser

import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core import eval_map
from mmdet.core.evaluation.mean_ap_visualize import map_roc_pr
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)

def voc_eval(result_file, dataset, iou_thr=0.5):
    det_results = mmcv.load(result_file)
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        bboxes = ann['bboxes']
        labels = ann['labels']
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
    
    dataset_name = dataset.CLASSES
    map_roc_pr(
        det_results,
        gt_bboxes,
        gt_labels,
        scale_ranges=None,
        dataset=dataset_name)


def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    test_dataset = build_dataset(cfg.data.test)
    voc_eval(args.result, test_dataset, args.iou_thr)


if __name__ == '__main__':
    main()
