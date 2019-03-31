import torch
import torch.nn as nn

class E1(nn.Module):
    def __init__(self, sep, size):
        super(E1, self).__init__()
        self.sep = sep
        self.size = size

        self.full = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, (512 - self.sep), 4, 2, 1),
            nn.InstanceNorm2d(512 - self.sep),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d((512 - self.sep), (512 - self.sep), 4, 2, 1),
            nn.InstanceNorm2d(512 - self.sep),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, net):
        net = self.full(net)
        net = net.view(-1, (512 - self.sep) * self.size * self.size)
        return net


class E2(nn.Module):
    def __init__(self, sep, size):
        super(E2, self).__init__()
        self.sep = sep
        self.size = size

        self.full = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, self.sep, 4, 2, 1),
            nn.InstanceNorm2d(self.sep),
            nn.LeakyReLU(0.2),
        )

    def forward(self, net):
        net = self.full(net)
        net = net.view(-1, self.sep * self.size * self.size)
        return net


class Decoder(nn.Module):
    def __init__(self, size):
        super(Decoder, self).__init__()
        self.size = size

        self.ss = nn.AvgPool2d(2, stride=2)

        self.ct1 = nn.ConvTranspose2d(515, 512, 4, 2, 1),
        self.in1 = nn.InstanceNorm2d(512),
        self.re1 = nn.ReLU(inplace=True),
        self.ct2 = nn.ConvTranspose2d(515, 256, 4, 2, 1),
        self.in2 = nn.InstanceNorm2d(256),
        self.re2 = nn.ReLU(inplace=True),
        self.ct3 = nn.ConvTranspose2d(259, 128, 4, 2, 1),
        self.in3 = nn.InstanceNorm2d(128),
        self.re3 = nn.ReLU(inplace=True),
        self.ct4 = nn.ConvTranspose2d(131, 64, 4, 2, 1),
        self.in4 = nn.InstanceNorm2d(64),
        self.re4 = nn.ReLU(inplace=True),
        self.ct5 = nn.ConvTranspose2d(67, 32, 4, 2, 1),
        self.in5 = nn.InstanceNorm2d(32),
        self.re5 = nn.ReLU(inplace=True),
        self.ct6 = nn.ConvTranspose2d(35, 3, 4, 2, 1),
        self.ta = nn.Tanh()

        # self.main = nn.Sequential(
        #     nn.ConvTranspose2d(512, 512, 4, 2, 1),
        #     nn.InstanceNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(512, 256, 4, 2, 1),
        #     nn.InstanceNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(256, 128, 4, 2, 1),
        #     nn.InstanceNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(128, 64, 4, 2, 1),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(64, 32, 4, 2, 1),
        #     nn.InstanceNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(32, 3, 4, 2, 1),
        #     nn.Tanh()
        # )

    def forward(self, net, true):
        net = net.view(-1, 512, self.size, self.size)
        net = self.main(net)

        T128 = self.ss(true)
        T64 = self.ss(T128)
        T32 = self.ss(T64)
        T16 = self.ss(T32)
        T8 = self.ss(T16)
        T4 = self.ss(T8)

        net = torch.cat([net, T4], dim=1)
        net = self.ct1(net)
        net = self.in1(net)
        net = self.re1(net)
        net = torch.cat([net, T8], dim=1)
        net = self.ct2(net)
        net = self.in2(net)
        net = self.re2(net)
        net = torch.cat([net, T16], dim=1)
        net = self.ct3(net)
        net = self.in3(net)
        net = self.re3(net)
        net = torch.cat([net, T32], dim=1)
        net = self.ct4(net)
        net = self.in4(net)
        net = self.re4(net)
        net = torch.cat([net, T64], dim=1)
        net = self.ct5(net)
        net = self.in5(net)
        net = self.re5(net)
        net = torch.cat([net, T128], dim=1)
        net = self.ct6(net)
        net = self.ta(net)
        return net


class Disc(nn.Module):
    def __init__(self, sep, size):
        super(Disc, self).__init__()
        self.sep = sep
        self.size = size

        self.classify = nn.Sequential(
            nn.Linear((512 - self.sep) * self.size * self.size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, net):
        net = net.view(-1, (512 - self.sep) * self.size * self.size)
        net = self.classify(net)
        net = net.view(-1)
        return net
