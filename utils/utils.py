import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch

def visualize_bb(opt, img, label, coord, save_path):

    MEAN = torch.tensor(opt.mean).cuda()
    STD = torch.tensor(opt.std).cuda()
    img = img * STD[:, None, None] + MEAN[:, None, None]
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1,2,0)) * 255
    pil_img = Image.fromarray(np.uint8(img))
    img = np.asarray(pil_img)
    xmin, ymin, xmax, ymax = coord[0], coord[1], coord[2], coord[3]
    w = xmax - xmin
    h = ymax - ymin
    img = cv2.resize(img, (w,h))
    w = label[0] * np.hypot(img.shape[0], img.shape[1])
    h = label[1] * np.hypot(img.shape[0], img.shape[1])
    theta = label[2] * np.pi
    theta = theta * 180/np.pi

    h = round(h)
    w = round(w)
    xc, yc = int(img.shape[1]/2), int(img.shape[0]/2)
    rect = (xc,yc), (h,w), 90-theta
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    bb_img = cv2.drawContours(img, [box], 0, (0,255,0), 1)
    plt.imsave(save_path, bb_img)

