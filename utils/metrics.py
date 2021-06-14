import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPoint
import cv2
import torch
import torch.nn as nn

def get_rbox(label):
    d1 = label[0] * 64
    d2 = label[1] * 64
    h = label[2] * np.hypot(64, 64)
    theta = np.arctan(d1/d2)
    theta = theta * 180/np.pi

    w = np.hypot(d1,d2)
    h = round(h)
    w = round(w)
    xc, yc = 32, 32
    rect = (xc,yc), (h,w), 90-theta
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

def skew_bbox_iou(box1, box2, GIoU=False):
    ft = torch.cuda.FloatTensor
    box1 = box1.detach().cpu().numpy()
    box2 = box2.detach().cpu().numpy()
    ious = []
    for i in range(len(box2)):
        r_b1 = get_rbox(box1[i])
        r_b2 = get_rbox(box2[i])

        ious.append(skewiou(r_b1, r_b2))

    return ft(ious)

def skewiou(box1, box2, mode='iou',return_coor = False):
    a=box1.reshape(4, 2)
    b=box2.reshape(4, 2)
    poly1 = Polygon(a).convex_hull
    poly2 = Polygon(b).convex_hull
    if not poly1.is_valid or not poly2.is_valid:
        print('formatting errors for boxes!!!! ')
        return 0
    if  poly1.area == 0 or  poly2.area  == 0 :
        return 0

    inter = Polygon(poly1).intersection(Polygon(poly2)).area
    if   mode == 'iou':
        union = poly1.area + poly2.area - inter
    elif mode =='tiou':
        union_poly = np.concatenate((a,b))
        union = MultiPoint(union_poly).convex_hull.area
        coors = MultiPoint(union_poly).convex_hull.wkt
    elif mode == 'giou':
        union_poly = np.concatenate((a,b))
        union = MultiPoint(union_poly).envelope.area
        coors = MultiPoint(union_poly).envelope.wkt
    elif mode== 'r_giou':
        union_poly = np.concatenate((a,b))
        union = MultiPoint(union_poly).minimum_rotated_rectangle.area
        coors = MultiPoint(union_poly).minimum_rotated_rectangle.wkt
    else:
        print('incorrect mode!')

    if union == 0:
        return 0
    else:
        if return_coor:
            return inter/union,coors
        else:
            return inter/union

def rmse_tensor(l1, l2):
    criterion = nn.MSELoss()
    loss = torch.sqrt(criterion(l1, l2))
    return loss

