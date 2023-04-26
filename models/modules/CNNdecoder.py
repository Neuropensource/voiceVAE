import torch.nn as nn


class ConvolutionalDecoder(nn.Module):

    def __init__(self, z_dim):
        super(ConvolutionalDecoder, self).__init__()

        self.mods = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, kernel_size=(27, 3), stride=(1, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 2), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 2), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 2), stride=(2, 2), padding=(1, 0), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=(4, 2), stride=(2, 2), padding=(1, 0), output_padding=[1, 1], bias=False)
        )

    def forward(self, z):
        if len(z.size()) == 1:
            z = z.unsqueeze(0)
        z = z.unsqueeze(2).unsqueeze(3)
        return self.mods(z)




class vanillaDecoder(nn.Module):

    def __init__(self, z_dim):
        super(vanillaDecoder, self).__init__()

        self.mods = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, kernel_size=(27, 3), stride=(1, 1), padding=(1, 0), bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 2), stride=(2, 1), padding=(1, 0), bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 2), stride=(2, 1), padding=(1, 0), bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 2), stride=(2, 2), padding=(1, 0), bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=(4, 2), stride=(2, 2), padding=(1, 0), output_padding=[1, 1], bias=False)
        )

    def forward(self, z):
        if len(z.size()) == 1:
            z = z.unsqueeze(0)
        z = z.unsqueeze(2).unsqueeze(3)
        return self.mods(z)