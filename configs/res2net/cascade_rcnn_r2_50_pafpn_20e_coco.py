_base_ = '../cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco.py'
model = dict(
    pretrained='F:\crh\mmdetection-2.1.0\checkpoints\\res2net50_v1b_26w_4s-3cf99910_mmdetv2.pth',
    backbone=dict(type='Res2Net', depth=50, scales=4, base_width=26),
	neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5))
