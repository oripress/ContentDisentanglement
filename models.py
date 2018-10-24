import torch
import torch.nn as nn


# class E1(nn.Module):
#     def __init__(self, sep, size):
#         super(E1, self).__init__()
#         self.sep = sep
#         self.size = size
#
#         self.first = nn.Sequential(
#             # nn.Conv2d(3, 16, 4, 2, 1),
#             # nn.InstanceNorm2d(16),
#             # nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(3, 32, 4, 2, 1),
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(32, 64, 4, 2, 1),
#             nn.InstanceNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, 4, 2, 1),
#             nn.InstanceNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, 4, 2, 1),
#             nn.InstanceNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, (512 - self.sep), 4, 2, 1),
#             nn.InstanceNorm2d(512 - self.sep),
#             nn.LeakyReLU(0.2, inplace=True),
#             # nn.Conv2d((512 - self.sep), (512 - self.sep), 4, 2, 1),
#             # nn.InstanceNorm2d(512 - self.sep),
#             # nn.LeakyReLU(0.2, inplace=True),
#         )
#
#
#     def forward(self, net):
#         net = self.first(net)
#         net = net.view(-1, (512 - self.sep) * self.size * self.size)
#         return net, net
#
#
# class E2(nn.Module):
#     def __init__(self, sep, size):
#         super(E2, self).__init__()
#         self.sep = sep
#         self.size = size
#
#         self.full = nn.Sequential(
#             # nn.Conv2d(3, 16, 4, 2, 1),
#             # nn.InstanceNorm2d(16),
#             # nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(3, 32, 4, 2, 1),
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(32, 64, 4, 2, 1),
#             nn.InstanceNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, 4, 2, 1),
#             nn.InstanceNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 128, 4, 2, 1),
#             nn.InstanceNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 128, 4, 2, 1),
#             nn.InstanceNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, self.sep, 4, 2, 1),
#             nn.InstanceNorm2d(self.sep),
#             nn.LeakyReLU(0.2),
#         )
#
#     def forward(self, net):
#         net = self.full(net)
#         net = net.view(-1, self.sep * self.size * self.size)
#         return net
#
#
# class Decoder(nn.Module):
#     def __init__(self, size):
#         super(Decoder, self).__init__()
#         self.size = size
#
#         self.first = nn.Sequential(
#             nn.ConvTranspose2d(512, 512, 4, 2, 1),
#             nn.InstanceNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(512, 256, 4, 2, 1),
#             nn.InstanceNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(258, 128, 4, 2, 1),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 32, 4, 2, 1),
#             # nn.InstanceNorm2d(32),
#             # nn.ReLU(inplace=True),
#             # nn.ConvTranspose2d(32, 3, 4, 2, 1),
#             # nn.InstanceNorm2d(16),
#             # nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(16, 3, 4, 2, 1),
#             nn.Tanh()
#         )
#
#
#     def forward(self, net, res):
#         net = net.view(-1, 512, self.size, self.size)
#         net = self.first(net)
#         return net


class E1(nn.Module):
    def __init__(self, sep, size):
        super(E1, self).__init__()
        self.sep = sep
        self.size = size

        self.first = nn.Sequential(
            # nn.Conv2d(3, 16, 4, 2, 1),
            # nn.InstanceNorm2d(16),
            # nn.LeakyReLU(0.2, inplace=True),
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
        )
        self.second = nn.Sequential(
            nn.Conv2d(256, (512 - self.sep), 4, 2, 1),
            nn.InstanceNorm2d(512 - self.sep),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d((512 - self.sep), (512 - self.sep), 4, 2, 1),
            nn.InstanceNorm2d(512 - self.sep),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.res = nn.Sequential(
            nn.Conv2d(256, 2, 1),
            # nn.InstanceNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, net):
        net = self.first(net)
        res = self.res(net)
        net = self.second(net)
        net = net.view(-1, (512 - self.sep) * self.size * self.size)
        return net, res


class E2(nn.Module):
    def __init__(self, sep, size):
        super(E2, self).__init__()
        self.sep = sep
        self.size = size

        self.full = nn.Sequential(
            # nn.Conv2d(3, 16, 4, 2, 1),
            # nn.InstanceNorm2d(16),
            # nn.LeakyReLU(0.2, inplace=True),
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

        self.first = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.second = nn.Sequential(
            nn.ConvTranspose2d(258, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            # nn.InstanceNorm2d(16),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Tanh()
        )


    def forward(self, net, res):
        net = net.view(-1, 512, self.size, self.size)
        net = self.first(net)
        net = torch.cat([net, res], dim=1)
        net = self.second(net)
        return net


class Disc(nn.Module):
    def __init__(self, sep, size):
        super(Disc, self).__init__()
        self.sep = sep
        self.size = size

        self.classify = nn.Sequential(
            nn.Linear(((512 - self.sep) * self.size * self.size) + 2 * 8 * 8, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, net):
        net = net.view(-1, ((512 - self.sep) * self.size * self.size) + 2 * 8 * 8)
        net = self.classify(net)
        net = net.view(-1)
        return net


class PatchDisc(nn.Module):
    def __init__(self):
        super(PatchDisc, self).__init__()
        self.full = nn.Sequential(
            # nn.Conv2d(3, 16, 4, 2, 1),
            # nn.InstanceNorm2d(16),
            # nn.LeakyReLU(0.2, inplace=True),
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
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, net):
        net = self.full(net)
        net = net.view(-1, 1)
        return net