import torch

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# config = "configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py"
# checkpoint = "work_dirs/faster_rcnn_r50_fpn_2x_coco/latest.pth"
# config = "configs/retinanet/retinanet_r50_fpn_2x_coco.py"
# checkpoint = "work_dirs/retinanet_r50_fpn_2x_coco/epoch_24.pth"
config = "configs/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco.py"
checkpoint = "work_dirs/cascade_rcnn_r50_fpn_20e_coco/epoch_20.pth"
# checkpoint = "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# build the model from a config file and a checkpoint file
img = "img/11.jpg"

model = init_detector(config, checkpoint, device=device)
# test a single image
result = inference_detector(model, img)
# show the results
show_result_pyplot(model, img, result, score_thr=0.5)
# RoIPool

