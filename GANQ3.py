from __future__ import print_function
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.autograd import grad

# %matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import dataset
import torchvision.datasets as dset
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


## adapated from : https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.htm
## with WGAN-GP objective


##Todo https://github.com/soumith/ganhacks
## for WGAN-GP change batch norm to instance normalisation
## Noisy and soft labels for discriminator
## Use SGD for discriminator and ADAM for generator

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# D(x)
def dcgan_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Norm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, f=32):
        super(Discriminator, self).__init__()

        # base nb of feature maps
        self.f = f

        def conv_block(in_size, out_size, k=4, s=2, p=1):
            return nn.Sequential(
                nn.Conv2d(in_size, out_size, k, s, p, bias=False),
                nn.InstanceNorm2d(out_size),  # No batch norm for WGAN-GP
                nn.LeakyReLU(0.2, inplace=True))

        self.main = nn.Sequential(
            nn.Conv2d(3, self.f, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            conv_block(self.f, self.f * 2),
            conv_block(self.f * 2, self.f * 4),
            conv_block(self.f * 4, self.f * 8),

            nn.Conv2d(self.f * 8, 1, 2, 1, 0, bias=False),
            # nn.Sigmoid()  # Remove Sigmoid for WGAN-GP objective
        )

    def forward(self, x):
        return self.main(x)


# G(z)
class Generator(nn.Module):

    def __init__(self, f=32):
        super(Generator, self).__init__()

        # base nb of feature maps
        self.f = f

        def dc_block(in_size, out_size, k=4, s=2, p=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_size, out_size, k, s, p, bias=False),
                nn.BatchNorm2d(out_size),
                nn.ReLU(True))

        self.main = nn.Sequential(
            dc_block(100, self.f * 8, 2, 1, 0),
            dc_block(self.f * 8, self.f * 4),
            dc_block(self.f * 4, self.f * 2),
            dc_block(self.f * 2, self.f),
            nn.ConvTranspose2d(self.f, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)


generator = Generator()
discriminator = Discriminator()
generator.cuda()
discriminator.cuda()


# Gradient Penalty : https://arxiv.org/pdf/1704.00028.pdf 4.0
# takes in one batch of real and fake data
def gradient_penalty(real, fake, factor=10):
    size = real.size()[0]

    # x_hat sampling
    alpha = torch.rand(size, 1, 1, 1, device=device, requires_grad=True)
    alpha = alpha.expand_as(real)
    x_hat = alpha * real.data + (1 - alpha) * fake.data

    # prediction
    D_x = Discriminator(x_hat)

    # https://discuss.pytorch.org/t/gradient-penalty-with-respect-to-the-network-parameters/11944/5
    gradients = grad(outputs=D_x, inputs=x_hat,
                     grad_outputs=torch.ones_like(D_x).to(device),
                     retain_graph=True, create_graph=True, only_inputs=True)[0]
    gradients = gradients.view(size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()

    return factor * penalty


# Generate image grid

# Train

from torch.optim import Adam
