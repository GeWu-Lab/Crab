import torch
import math
import numpy as np

def bbox_overlaps_ciou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - iou
        alpha = v / (S + v)
        w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    cious = iou - (u + alpha * ar)
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    if exchange:
        cious = cious.T
    return cious


def intersection_over_union(box1, box2, wh=False):
    """
    计算IoU（交并比）
    :param box1: bounding box1
    :param box2: bounding box2
    :param wh: 坐标的格式是否为（x,y,w,h）
    :return:计算结果
    """
    if not wh:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)
    # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = max([xmin1, xmin2])
    yy1 = max([ymin1, ymin2])
    xx2 = min([xmax1, xmax2])
    yy2 = min([ymax1, ymax2])
    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    inter_area = (max([0, xx2 - xx1])) * (max([0, yy2 - yy1]))  # 计算交集面积
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)  # 计算交并比
    return iou


def c_iou(rec1, rec2):
    """
    计算CIoU
    :param rec1: bounding box1（xmin, ymin, xmax, ymax）格式
    :param rec2: bounding box2（xmin, ymin, xmax, ymax）格式
    :return: 计算结果
    """
    # xmin, ymin, xmax, ymax格式
    xmin1, ymin1, xmax1, ymax1 = rec1
    xmin2, ymin2, xmax2, ymax2 = rec2
    iou = intersection_over_union(rec1, rec2)
    # 中心点距离平方
    center1 = ((xmin1 + xmax1) / 2, (ymin1 + ymax1) / 2)
    center2 = ((xmin2 + xmax2) / 2, (ymin2 + ymax2) / 2)
    d_center2 = (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
    # 最小闭包矩形对角线长度平方
    corner1 = (min(xmin1, xmax1, xmin2, xmax2), min(ymin1, ymax1, ymin2, ymax2))
    corner2 = (max(xmin1, xmax1, xmin2, xmax2), max(ymin1, ymax1, ymin2, ymax2))
    d_corner2 = (corner1[0] - corner2[0]) ** 2 + (corner1[1] + corner2[1]) ** 2
    w1, h1 = xmax1 - xmin1, ymax1 - ymin1
    w2, h2 = xmax2 - xmin2, ymax2 - ymin2
    # 度量长宽比的参数
    v = 4 * (np.arctan(w1 / h1) - np.arctan(w2 / h2)) ** 2 / (np.pi ** 2)
    alpha = v / (1 - iou + v)
    ciou = iou - d_center2 / d_corner2 - alpha * v
    return ciou


