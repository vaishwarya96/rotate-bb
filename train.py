from options.train_options import TrainOptions
import sys
import os
from utils.dataset import LoadDataset
import torch.nn as nn
from torch.optim import Adam
import torch
from models.model import LeNet5
import torch.utils.data as data
from val import val
from utils.utils import visualize_bb

#get the arguments
parser = TrainOptions().print_parse()
print(' '.join(sys.argv))
opt = parser.parse_args()

#Path to save the trained models
model_path = opt.checkpoint_dir
if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=True)
model_path = os.path.join(model_path, 'network.pth')

num_epochs = opt.num_epochs

train_data = LoadDataset(opt, opt.train_image_paths, opt.train_label_paths)
train_data_loader = data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=4)

model = LeNet5()
model = model.cuda()

best_iou = 0.0

optimizer = Adam(model.parameters(), lr=opt.lr)
loss = nn.MSELoss()

for epoch in range(num_epochs):
    for iter, (img, label, coord) in enumerate(train_data_loader):

        img = img.cuda()
        label = label.type(torch.FloatTensor).cuda()
        optimizer.zero_grad()
        logits = model(img)
        loss_value = 5*loss(logits, label)
        loss_value.backward()
        optimizer.step()

        if epoch == 0 and iter==0:
            torch.save(model.state_dict(), model_path)
            print("Model Saved")

        iou = val(opt, model)
        print(iou)
        if iou > best_iou:
            best_iou = iou
            print("Best IOU is %f" %(best_iou))
            torch.save(model.state_dict(), model_path)
            print("Model Saved")

    print("Epoch: %d, Loss: %f" %(epoch, loss_value.item()))

print("Training Over")


