# dataset settings
dataset_type = 'RDDDataset'
data_root = '/home/training/datasets/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# img_scale = [(640, 640), (720,720), (800, 800), (880, 880)]
# img_scale = [(320, 320), (480,480), (560,560), (640, 640), (720,720), (800, 800), (880, 880)]
img_scale = [ (320,320), (416,416),  (640, 640),  (800, 800), (960, 960), (1280, 1280)]
train_pipeline = [  
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, multiscale_mode='value', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    # dict(type='Rotate')
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=[(640, 640),(720, 720),(960,960)],
        # img_scale = [(640, 640), (720,720), (800, 800), (880, 880)],
        img_scale = (640, 640),
        # img_scale = (1280, 1280),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'pothole/train/pothole.txt', data_root + 'pothole1/train/train.txt', 
            ],
            img_prefix=[data_root + 'pothole/train/', data_root + 'pothole1/train/', 
                        ],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=[data_root + 'pothole1/valid/valid.txt',],
        img_prefix=[data_root + 'pothole1/valid/',],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=[data_root + 'pothole/test.txt',data_root + 'pothole1/test.txt',],
        img_prefix=[data_root + 'pothole/test/',data_root + 'pothole1/test',],
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
