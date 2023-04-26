import torch.nn as nn


class ConvolutionalEncoder(nn.Module):
    
    def __init__(self, z_dim):
        super(ConvolutionalEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=[6, 3], stride=[2, 2], padding=(1, 0), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=[6, 2], stride=[2, 2], padding=(1, 0), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=[6, 2], stride=[2, 1], padding=(1, 0), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=[6, 2], stride=[2, 1], padding=(1, 0), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(512, 7, kernel_size=[6, 2], stride=[1, 1], padding=(1, 0), bias=False),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(280, z_dim*2)

    def forward(self, x):
        conv = self.layers(x)
        h1 = conv.view(-1, 280)
        h2 = self.fc1(h1)
        return h2



class vanillaEncoder(nn.Module):
    
    def __init__(self, z_dim):
        super(vanillaEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=[6, 3], stride=[2, 2], padding=(1, 0), bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=[6, 2], stride=[2, 2], padding=(1, 0), bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=[6, 2], stride=[2, 1], padding=(1, 0), bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=[6, 2], stride=[2, 1], padding=(1, 0), bias=False),
            nn.ReLU(),
            nn.Conv2d(512, 7, kernel_size=[6, 2], stride=[1, 1], padding=(1, 0), bias=False),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(280, z_dim*2)

    def forward(self, x):
        conv = self.layers(x)
        h1 = conv.view(-1, 280)
        h2 = self.fc1(h1)
        return h2