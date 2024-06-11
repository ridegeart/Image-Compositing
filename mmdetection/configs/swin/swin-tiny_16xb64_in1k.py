_base_ = [
    '../_base_/models/tiny_224.py',
    '../_base_/datasets/crater_w12_3.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]
pretrained = 'checkpoints/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    )
# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))