import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPoint
import cv2
import torch
import torch.nn as nn

def get_rbox(opt,label):
    w = label[0] * np.hypot(opt.image_size, opt.image_size)
    h = label[1] * np.hypot(opt.image_size, opt.image_size)
    theta = label[2] * np.pi
    theta = theta * 180/np.pi

    h = round(h)
    w = round(w)
    xc, yc = int(opt.image_size/2), int(opt.image_size/2)
    rect = (xc,yc), (h,w), 90-theta
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

def skew_bbox_iou(opt, box1, box2, GIoU=False):
    ft = torch.cuda.FloatTensor
    box1 = box1.detach().cpu().numpy()
    box2 = box2.detach().cpu().numpy()
    ious = []
    for i in range(len(box2)):
        r_b1 = get_rbox(opt, box1[i])
        r_b2 = get_rbox(opt, box2[i])

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

