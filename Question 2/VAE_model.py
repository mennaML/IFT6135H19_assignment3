import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=5)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=5, padding=4)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=2)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, padding=2)
        self.conv7 = nn.Conv2d(16, 1, kernel_size=3, padding=2)
        
        self.ELU = nn.ELU()
        self.AvgPool2d = nn.AvgPool2d(kernel_size=2, stride=2)
        self.enc_fc_mu = nn.Linear(256, 100)
        self.enc_fc_logvar = nn.Linear(256, 100)
        self.dec_fc = nn.Linear(100, 256)

        self.upsamplingx2 = F.interpolate#nn.UpsamplingBilinear2d(scale_factor=2)

    def encode(self, x):
        x = self.conv1(x)
        x = self.ELU(x)
        x = self.AvgPool2d(x)
        x = self.conv2(x)
        x = self.ELU(x)
        x = self.AvgPool2d(x)
        x = self.conv3(x)
        x = self.ELU(x)
        return self.enc_fc_mu(x.squeeze()), self.enc_fc_logvar(x.squeeze())

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std    

    def decode(self, z):

        # z 100x1
        x = self.dec_fc(z)
        x = self.ELU(x)
        
        x = x.view(x.size(0), x.size(1), 1, 1)
        
        x = self.conv4(x)
        x = self.ELU(x)

        x = self.upsamplingx2(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv5(x)
        x = self.ELU(x)

        x = self.upsamplingx2(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv6(x)
        x = self.ELU(x)

        x = self.conv7(x)
        # x 28x28
        return torch.sigmoid(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
