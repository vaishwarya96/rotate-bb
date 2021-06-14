import sys
import os
from utils.dataset import LoadDataset
from models.model import LeNet5
import torch
import torch.nn as nn
from utils.metrics import rmse_tensor
from utils.utils import visualize_bb
import torch.utils.data as data
import numpy as np
from utils.metrics import skew_bbox_iou

def val(opt, model):

    result_path = opt.result_dir
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)

    test_data = LoadDataset(opt, opt.test_image_paths, opt.test_label_paths)
    test_data_loader = data.DataLoader(test_data, batch_size = 2, shuffle=False, num_workers=4)

    model = model.cuda()
    model.eval()
    
    ious = []
    tot_rmse = 0
    with torch.no_grad():
        for iter, (img, label, coord) in enumerate(test_data_loader):
            img = img.cuda()
            label = label.type(torch.FloatTensor).cuda()
            logits = model(img)
            
            #ious.append([1])
            ious.append(skew_bbox_iou(label, logits).detach().cpu().numpy())
            #print(logits, label)
            tot_rmse += rmse_tensor(logits, label)

            img = img[0]
            logit = logits[0]
            coord = coord[0].numpy()
            logit = logit.detach().cpu().numpy()
            save_path = os.path.join(opt.result_dir, str(iter)+'.png')
            if not os.path.exists(opt.result_dir):
                os.makedirs(opt.result_dir, exist_ok=True)
            visualize_bb(img, logit, coord, save_path)


    return np.mean(ious)


