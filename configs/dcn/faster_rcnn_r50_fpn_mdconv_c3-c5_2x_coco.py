_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
