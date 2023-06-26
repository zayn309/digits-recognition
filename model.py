import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3)  # 1 input channel, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(64, 64, 3)  # 64 input channels, 64 output channels, 3x3 kernel
        self.maxpool1 = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)  # 64 input channels, 128 output channels, 3x3 kernel
        self.conv4 = nn.Conv2d(128, 128, 3)  # 128 input channels, 128 output channels, 3x3 kernel
        self.maxpool2 = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        self.batchnorm2 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3)  # 128 input channels, 256 output channels, 3x3 kernel
        self.maxpool3 = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        self.batchnorm3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256, 512)  # Fully connected layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)  # Output layer with 10 classes (0-9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.maxpool3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        
        return x
