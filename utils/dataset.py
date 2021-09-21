import torch.utils.data as data
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt

class LoadDataset(data.Dataset):
    def __init__(self, opt, image_dir, label_dir):
        self.opt = opt
        self.image_dir = image_dir
        self.label_dir = label_dir
        self. image_paths, self.label_paths = self.get_paths(self.image_dir, self.label_dir)
        self.mean = self.opt.mean
        self.std = self.opt.std

        self.images = []
        self.label = []
        self.coord = []

        for idx in range(len(self.image_paths)):
            img = cv2.imread(self.image_paths[idx])
            label_file = self.label_paths[idx]
            with open(label_file, 'r') as f:
                l = np.array([x.split() for x in f.read().splitlines()])
            for i in range(l.shape[0]):
                x_min, y_min, x_max, y_max, a, w, h = max(int(l[i][0]),0), max(int(l[i][1]),0), min(int(l[i][2]),img.shape[0]), min(int(l[i][3]),img.shape[1]), float(l[i][4]), float(l[i][5]), float(l[i][6])
                bb_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
                bb_img = cv2.resize(bb_img, (self.opt.image_size, self.opt.image_size))
                theta = a/np.pi
                w = w/np.hypot(self.opt.image_size, self.opt.image_size)
                h = h/np.hypot(self.opt.image_size, self.opt.image_size)
                self.images.append(bb_img)
                self.label.append([w, h, theta])
                self.coord.append([x_min, y_min, x_max, y_max])

        assert len(self.images) == len(self.label), "Number of images and labels are different"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img = self.images[index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        label = self.label[index]
        coord = self.coord[index]
        img_trans = self.img_transform()
        img = img_trans(img)
        return img, torch.tensor(label), torch.tensor(coord)

    def img_transform(self):
        trans = []
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=self.mean,std=self.std))
        return transforms.Compose(trans)


    def get_paths(self, image_dir, label_dir):
        image_paths = glob.glob(image_dir+'/*')
        label_paths = []
        for i in image_paths:
            basename = os.path.basename(i)
            filename = os.path.splitext(basename)[0]
            label_paths.append(os.path.join(label_dir, filename+'.txt'))

        return image_paths, label_paths




