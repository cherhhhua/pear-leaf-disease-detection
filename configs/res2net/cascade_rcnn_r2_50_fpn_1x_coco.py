_base_ = '../cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
model = dict(
    pretrained='F:\crh\mmdetection-2.1.0\checkpoints\\res2net50_v1b_26w_4s-3cf99910_mmdetv2.pth',
    backbone=dict(type='Res2Net', depth=50, scales=4, base_width=26))
