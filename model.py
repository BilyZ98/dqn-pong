import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, height=84, width=84, inchannel=4, output=14):
        super(DQN, self).__init__()
    
    
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size)) // stride + 1

        self.conv1 = nn.Conv2d(inchannel, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        
        self.linear = nn.Linear(convw * convh * 32, 512)
        self.head = nn.Linear(512, output)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.linear(x.view(x.size(0), -1)))
        x = self.head(x)
        return x
