import os, sys
import logging

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../FedML")))
from fedml_api.model.cv.batchnorm_utils import SynchronizedBatchNorm2d

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, sync_bn=False):
        super().__init__()

        if sync_bn == True:
            BatchNorm2d = SynchronizedBatchNorm2d
        else:
            BatchNorm2d = nn.BatchNorm2d

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class UnetPlusPlus(nn.Module):
    def __init__(self, n_classes, nInputChannels=3, sync_bn=False, **kwargs):
        super().__init__()

        channels = 64
        nb_filter = [channels, channels * 2, channels * 4, channels * 8, channels * 16]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(nInputChannels, nb_filter[0], nb_filter[0], sync_bn)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], sync_bn)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], sync_bn)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], sync_bn)
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4], sync_bn)

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0], sync_bn)
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1], sync_bn)
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2], sync_bn)
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3], sync_bn)

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0], sync_bn)
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1], sync_bn)
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2], sync_bn)

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0], sync_bn)
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1], sync_bn)

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0], sync_bn)

        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

if __name__ == "__main__":
    image = torch.randn(16,3,512,512)
    model = UnetPlusPlus(n_classes=21)
    with torch.no_grad():
        output = model.forward(image)
    print(output.size())