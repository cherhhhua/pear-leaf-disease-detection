import time

import mmcv
import numpy as np
import slidingwindow as sw
import cv2
import torch
from PIL import Image

# import torchvision.ops.nms
from mmdet.apis import init_detector, inference_detector, show_result_pyplot


def sliding_crop_test(imgpath, config, checkpoint, device):
    model = init_detector(config, checkpoint, device=device)
    # test a single image
    data = mmcv.imread(imgpath)
    # image = cv.cvtColor(data, cv.COLOR_BGR2RGB)
    windows_bbox = []

    windows = sw.generate(data, sw.DimOrder.HeightWidthChannel, 500, 0.15)
    # print(windows)

    for index in range(len(windows)):
        subset = data[windows[index].indices()]
        # img = Image.fromarray(subset)
        w1, h1 = windows[index].getRect()[0:2]
        result = inference_detector(model, subset)
        result_1 = show_result_pyplot(model, subset, result, score_thr=0.5)
        for i in result_1:
            left, top, right, bottom, score, label = i
            left = left + w1
            top = top + h1
            right = right + w1
            bottom = bottom + h1
            windows_bbox.append([left, top, right, bottom, score, label])

    return data, windows_bbox


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    # 所有图片的坐标信息，字典形式储存？？
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 计算出所有图片的面积
    order = scores.argsort()[::-1]  # 图片评分按升序排序

    keep = []  # 用来存放最后保留的图片的相应评分
    while order.size > 0:
        i = order[0]  # i 是还未处理的图片中的最大评分
        keep.append(i)  # 保留改图片的值
        # 矩阵操作，下面计算的是图片i分别与其余图片相交的矩形的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算出各个相交矩形的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠比例
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 只保留比例小于阙值的图片，然后继续处理
        inds = np.where(ovr <= thresh)[0]
        indsd= inds+1
        order = order[inds + 1]

    return keep


def soft_nms(dets, sigma=0.5, Nt=0.5, method=2, threshold=0.1):
    box_len = len(dets)  # box的个数
    for i in range(box_len):
        tmpx1, tmpy1, tmpx2, tmpy2, ts = dets[i, 0], dets[i, 1], dets[i, 2], dets[i, 3], dets[i, 4]
        max_pos = i
        max_scores = ts

        # get max box
        pos = i + 1
        while pos < box_len:
            if max_scores < dets[pos, 4]:
                max_scores = dets[pos, 4]
                max_pos = pos
            pos += 1

        # add max box as a detection
        dets[i, :] = dets[max_pos, :]

        # swap ith box with position of max box
        dets[max_pos, 0] = tmpx1
        dets[max_pos, 1] = tmpy1
        dets[max_pos, 2] = tmpx2
        dets[max_pos, 3] = tmpy2
        dets[max_pos, 4] = ts

        # 将置信度最高的 box 赋给临时变量
        tmpx1, tmpy1, tmpx2, tmpy2, ts = dets[i, 0], dets[i, 1], dets[i, 2], dets[i, 3], dets[i, 4]

        pos = i + 1
        # NMS iterations, note that box_len changes if detection boxes fall below threshold
        while pos < box_len:
            x1, y1, x2, y2 = dets[pos, 0], dets[pos, 1], dets[pos, 2], dets[pos, 3]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)

            iw = (min(tmpx2, x2) - max(tmpx1, x1) + 1)
            ih = (min(tmpy2, y2) - max(tmpy1, y1) + 1)
            if iw > 0 and ih > 0:
                overlaps = iw * ih
                ious = overlaps / ((tmpx2 - tmpx1 + 1) * (tmpy2 - tmpy1 + 1) + area - overlaps)
                coincide = overlaps / min((tmpx2 - tmpx1 + 1) * (tmpy2 - tmpy1 + 1), area)
                # if min((tmpx2 - tmpx1 + 1) * (tmpy2 - tmpy1 + 1), area) == area:
                #     coincide = overlaps / min((tmpx2 - tmpx1 + 1) * (tmpy2 - tmpy1 + 1), area)
                # else:
                #     coincide = 0
                if method == 1:  # 线性
                    if ious > Nt:
                        weight = 1 - ious
                    else:
                        weight = 1
                elif method == 2:  # gaussian
                    # weight = np.exp(-(ious ** 2 / sigma))
                    weight = np.exp(-(ious ** 2 + coincide ** 2) / sigma)
                else:  # original NMS
                    if ious > Nt:
                        weight = 0
                    else:
                        weight = 1

                # 赋予该box新的置信度
                dets[pos, 4] = weight * dets[pos, 4]

                # 如果box得分低于阈值thresh，则通过与最后一个框交换来丢弃该框
                if dets[pos, 4] < threshold:
                    dets[pos, 0] = dets[box_len - 1, 0]
                    dets[pos, 1] = dets[box_len - 1, 1]
                    dets[pos, 2] = dets[box_len - 1, 2]
                    dets[pos, 3] = dets[box_len - 1, 3]
                    dets[pos, 4] = dets[box_len - 1, 4]

                    box_len = box_len - 1
                    pos = pos - 1
            pos += 1

    pick = [i for i in range(box_len)]
    return pick


def draw_rectangle(data, bbox1, save_path, save_name):
    dets = np.array(bbox1)
    # pick = nms(dets, thresh=0.5)
    pick1 = py_cpu_nms(dets, 0.5)
    print(len(pick1))
    for i in dets[pick1]:
        left, top, right, bottom, score, label = i.astype(np.int32)
        if label == 0:
            cv2.rectangle(data,
                          (int(left), int(top)),
                          (int(right), int(bottom)),
                          (0, 0, 255), 2)
        if label == 1:
            cv2.rectangle(data,
                          (int(left), int(top)),
                          (int(right), int(bottom)),
                          (255, 0, 0), 2)
        if label == 2:
            cv2.rectangle(data,
                          (int(left), int(top)),
                          (int(right), int(bottom)),
                          (0, 255, 0), 2)
        if label == 3:
            cv2.rectangle(data,
                          (int(left), int(top)),
                          (int(right), int(bottom)),
                          (255, 255, 0), 2)

    cv2.imwrite(save_path + save_name, data)


if __name__ == "__main__":
    config = "configs/cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco.py"
    checkpoint = "work_dirs/cascade_rcnn_r101_fpn_20e_coco/epoch_20.pth"
    # config = "configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
    # checkpoint = "work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_12.pth"

    # config = "configs/retinanet/retinanet_r50_fpn_2x_coco.py"
    # checkpoint = "work_dirs/retinanet_r50_fpn_2x_coco/epoch_24.pth"
    # checkpoint = "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()
    image, bbox = sliding_crop_test("img/2058.jpg", config, checkpoint, device)
    print("First time: ", time.time() - start_time)
    draw_rectangle(image, bbox, "result/", "2858_cascade_101.jpg")
    print("End time: ", time.time() - start_time)
