import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch
import os
from torchsummary import summary
import torchsnooper
import math

patch_size = 7
batch_size = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn(batch_size, 1, 103, patch_size, patch_size, device=device)
# -----------------------自加在构建网络的情况下获得维度---------------------------
# @torchsnooper.snoop()

# class Net(nn.Module):
#     @staticmethod
#     def weight_init(m):
#         if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
#             init.xavier_uniform_(m.weight.data)
#             init.constant_(m.bias.data, 0)
#
#     def _get_final_flattened_size(self):
#         with torch.no_grad():
#             x = torch.zeros((1, 1, 103,
#                              patch_size, patch_size),device=device)
#             x = self.pool1(self.conv1(x))
#             x = self.pool2(self.conv2(x))
#             x = self.conv3(x)
#             _, t, c, w, h = x.size()
#         return t * c * w * h
#
#     def __init__(self):
#         super(Net,self).__init__()
#         self.conv1 = nn.Conv3d(1, 32, (32, 4, 4), padding=(1, 1, 1)).cuda()
#         self.conv2 = nn.Conv3d(32, 2*32, (32, 5, 5), padding=(1, 1, 1)).cuda()
#         self.conv3 = nn.Conv3d(2*32, 4*32, (32, 3, 3), padding=(1, 0, 0)).cuda()
#         self.pool1 = nn.MaxPool3d((1,2,2), stride = (1,2,2)).cuda()
#         self.pool2 = nn.MaxPool3d((1,2,2), stride = (1,2,2)).cuda()
#
#         self.features_size = self._get_final_flattened_size()
#
#         self.fc = nn.Linear(self.features_size, 10).cuda()
#
#         self.apply(self.weight_init)
#
#     def forward(self,x):
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = F.relu(self.conv3(x))
#         x = x.view(-1, self.features_size)
#         x = self.fc(x)
#         return x
class OwnNetTest(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    遥感图像分类的三维深度学习方法
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5):
        super(OwnNetTest, self).__init__()
        # The first layer: (3,3,3) kernel, stride = 1 , number of neurons = 20
        self.patch_size = patch_size
        self.input_channels = input_channels

        # ================================multi-scale================================
        self.conv1 = nn.Conv3d(
            1, 20, (11, 3, 3), stride=(3, 1, 1), padding=(0, 0, 0))
        # 维持原有维度不变
        self.conv2_1 = nn.Conv3d(20, 20, (1, 1, 1), padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(20, 20, (3, 1, 1), padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(20, 20, (5, 1, 1), padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(20, 20, (11, 1, 1), padding=(5, 0, 0))
        # ================================multi-scale================================

        # =======================LocalResponseNorm for residual======================
        self.lrn = nn.LocalResponseNorm(72)       # 待改
        # =======================LocalResponseNorm for residual======================

        self.conv4 = nn.Conv3d(
            20, 35, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0))     # 3D convolution

        # residual block
        self.conv5 = nn.Conv3d(
            35, 35, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))     # 3D convolution
        self.conv6 = nn.Conv3d(
            35, 35, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))     # 3D convolution
        self.conv7 = nn.Conv3d(
            35, 35, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))     # 3D convolution

        self.conv8 = nn.Conv3d(
            35, 35, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))     # 3D convolution

        self.conv9 = nn.Conv3d(
            35, 35, (3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))     # 1D convolution


        self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    # 待改
    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = F.relu(self.conv1(x))           # 101 x 5 x 5

            # ================================multi-scale================================
            x = self.conv2_1(x)              # 51 x 5 x 5
            # ================================multi-scale================================

            x = F.relu(self.conv4(x))           # 51 x 3 x 3

            # residual clock
            x = F.relu(self.conv5(x))           # 51 x 3 x 3
            # x_res = self.conv6(x_res)               # 51 x 3 x 3
            # x_res = F.relu(self.conv7(x_res))       # 51 x 3 x 3
            # x = F.relu(x + x_res)                   # 51 x 3 x 3

            x = F.relu(self.conv8(x))               # 51 x 3 x 3

            # 1D convolution
            x = F.relu(self.conv9(x))           # 51 x 3 x 3

            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):

        x = F.relu(self.conv1(x))  # 101 x 5 x 5

        # # ================================multi-scale================================
        x2_1 = self.conv2_1(x)          # 51 x 5 x 5
        x2_2 = self.conv2_2(x)          # 51 x 5 x 5
        x2_3 = self.conv2_3(x)          # 51 x 5 x 5
        x2_4 = self.conv2_4(x)          # 51 x 5 x 5
        x = x2_1 + x2_2 + x2_3 + x2_4   # 51 x 5 x 5
        # # ================================multi-scale================================

        # LocalResponseNorm
        x = F.relu(self.lrn(x))  # 51 x 5 x 5

        x = F.relu(self.conv4(x))  # 51 x 3 x 3

        # residual clock
        x_res = F.relu(self.conv5(x))  # 51 x 3 x 3
        x_res = self.conv6(x_res)  # 51 x 3 x 3
        x_res = F.relu(self.conv7(x_res))  # 51 x 3 x 3
        x = F.relu(x + x_res)  # 51 x 3 x 3

        x = F.relu(self.conv8(x))  # 51 x 3 x 3

        # # 1D convolution
        x = F.relu(self.conv9(x))  # 51 x 3 x 3

        x = x.view(-1, self.features_size)

        x = self.dropout(x)
        x = self.fc(x)
        return x

# -------------------------自加网络---------------------------
net = OwnNetTest(input_channels=103, n_classes=10)
net.to(device)

# print(net.to(device))

summary(net.to(device), (1, 103, patch_size, patch_size), device=device)
# -----------------------自加在构建网络的情况下获得维度---------------------------
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv3d-1         [-1, 20, 31, 5, 5]           2,000
#
#             Conv3d-2         [-1, 20, 31, 5, 5]             420
#             Conv3d-3         [-1, 20, 31, 5, 5]           1,220
#             Conv3d-4         [-1, 20, 31, 5, 5]           2,020
#             Conv3d-5         [-1, 20, 31, 5, 5]           4,420
#
#  LocalResponseNorm-6         [-1, 20, 31, 5, 5]               0
#
#             Conv3d-7         [-1, 35, 31, 3, 3]          18,935
#
#             Conv3d-8         [-1, 35, 31, 3, 3]          33,110
#             Conv3d-9         [-1, 35, 31, 3, 3]          33,110
#            Conv3d-10         [-1, 35, 31, 3, 3]          33,110
#
#            Conv3d-11         [-1, 35, 31, 3, 3]          33,110
#
#            Conv3d-12         [-1, 35, 31, 3, 3]           3,710
#
#           Dropout-13                 [-1, 1085]               0
#            Linear-14                   [-1, 10]          10,860
# ================================================================
# Total params: 176,025
# Trainable params: 176,025
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.02
# Forward/backward pass size (MB): 1.16
# Params size (MB): 0.67
# Estimated Total Size (MB): 1.86
# ----------------------------------------------------------------