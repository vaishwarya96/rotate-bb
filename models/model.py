import torch.nn as nn
import torch

class LeNet5(nn.Module):
    
    def __init__(self):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(            
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
        nn.BatchNorm2d(6),
        nn.ReLU(),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.AvgPool2d(kernel_size=2),
        nn.Conv2d(in_channels=64, out_channels=120, kernel_size=5, stride=1),
        nn.BatchNorm2d(120),
        nn.AvgPool2d(kernel_size=2)
        )

        self.regresser = nn.Sequential(
                nn.Linear(in_features=120*11*11, out_features=84),
                #nn.BatchNorm2d(84),
                #nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=84, out_features=3),
                )
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.regresser(x)
        logits[:, :2] = nn.Sigmoid()(logits[:, :2])
        logits[:, 2:] = nn.Tanh()(logits[:, 2:])
        return logits
