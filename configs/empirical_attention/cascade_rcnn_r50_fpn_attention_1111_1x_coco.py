_base_ = '../cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(plugins=[
        dict(
            cfg=dict(
                type='GeneralizedAttention',
                spatial_range=-1,
                num_heads=8,
                attention_type='1111',
                kv_stride=2),
            stages=(False, False, True, True),
            position='after_conv2')
    ]))
test_cfg = dict(
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='soft_nms', iou_thr=0.5),
        max_per_img=100))

