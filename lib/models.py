#!/usr/bin/env python3
""" Pytorch Face Swapping Models """
import torch
from torch import nn

class Upscale(nn.Module):
    """ Upscale block to double the width/height from depth.  """
    def __init__(self, size):
        super().__init__()
        self.conv = nn.Conv2d(size * 2, size * 2 * 2, 3, 1, padding="same")
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        """ Upscale forward pass """
        x = self.conv(x)
        x = self.shuffle(x)
        return x

class OriginalEncoder(nn.Module):
    """ Face swapping encoder

    Shared to create encodings for both the faces.
    """
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(.1)
        self.conv_tower = nn.Sequential(nn.Conv2d(3, 128, 5, 2, padding=2),
                                        self.activation,
                                        nn.Conv2d(128, 256, 5, 2, padding=2),
                                        self.activation,
                                        nn.Conv2d(256, 512, 5, 2, padding=2),
                                        self.activation,
                                        nn.Conv2d(512, 1024, 5, 2, padding=2),
                                        self.activation)
        self.dense1 = nn.Linear(4 * 4 * 1024, 1024)
        self.dense2 = nn.Linear(1024, 4 * 4 * 1024)
        self.flatten = nn.Flatten()
        self.upscale = Upscale(512)

    def forward(self, x):
        """ Encoder forward pass """
        batch_size = x.shape[0]
        x = self.conv_tower(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = torch.reshape(x, [batch_size, 1024, 4, 4])
        x = self.upscale(x)
        x = self.activation(x)
        return x

class OriginalDecoder(nn.Module):
    """ Face swapping decoder

    An instance for each face to decode the shared encodings.
    """
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(.1)
        self.upscale_tower = nn.Sequential(Upscale(256),
                                           self.activation,
                                           Upscale(128),
                                           self.activation,
                                           Upscale(64),
                                           self.activation)
        self.output = nn.Conv2d(64, 3, 5, padding="same")
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        """ Decoder forward pass """
        x = self.upscale_tower(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x
