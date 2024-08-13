import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=4,
            stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)
        self.layer3 = nn.Linear(64*5*5, 128)
        self.layer4 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.layer4(x)
        x = self.softmax(x)
        return x
