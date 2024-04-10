import torch
from torch.nn import Module, Conv2d, MaxPool2d, ConvTranspose2d, BatchNorm2d, ReLU, Sigmoid

class UNet(Module):
    """
    Simple UNet model to perform image segmentation
    input: (N, 3, W, H)
    output: (N, 3, W, H)
    """

    def __init__(self, in_channels=3, out_channels=3, features=32):
        super(UNet, self).__init__()
        self.input = (in_channels, 256, 256) # (C, W, H)
        # Encoder
        self.conv1 = Conv2d(in_channels, features, 3, 1, 1)
        self.bn1 = BatchNorm2d(features)
        self.conv2 = Conv2d(features, features, 3, 1, 1)
        self.bn2 = BatchNorm2d(features)
        self.conv3 = Conv2d(features, features*2, 3, 1, 1)
        self.bn3 = BatchNorm2d(features*2)
        self.conv4 = Conv2d(features*2, features*2, 3, 1, 1)
        self.bn4 = BatchNorm2d(features*2)
        self.conv5 = Conv2d(features*2, features*4, 3, 1, 1)
        self.bn5 = BatchNorm2d(features*4)
        self.conv6 = Conv2d(features*4, features*4, 3, 1, 1)
        self.bn6 = BatchNorm2d(features*4)
        self.conv7 = Conv2d(features*4, features*8, 3, 1, 1)
        self.bn7 = BatchNorm2d(features*8)
        self.conv8 = Conv2d(features*8, features*8, 3, 1, 1)
        self.bn8 = BatchNorm2d(features*8)
        self.conv9 = Conv2d(features*8, features*16, 3, 1, 1)
        self.bn9 = BatchNorm2d(features*16)
        self.conv10 = Conv2d(features*16, features*16, 3, 1, 1)
        self.bn10 = BatchNorm2d(features*16)

        # Decoder
        self.upconv1 = ConvTranspose2d(features*16, features*8, 2, 2)
        self.conv11 = Conv2d(features*16, features*8, 3, 1, 1)
        self.bn11 = BatchNorm2d(features*8)
        self.conv12 = Conv2d(features*8, features*8, 3, 1, 1)
        self.bn12 = BatchNorm2d(features*8)
        self.upconv2 = ConvTranspose2d(features*8, features*4, 2, 2)
        self.conv13 = Conv2d(features*8, features*4, 3, 1, 1)
        self.bn13 = BatchNorm2d(features*4)
        self.conv14 = Conv2d(features*4, features*4, 3, 1, 1)
        self.bn14 = BatchNorm2d(features*4)
        self.upconv3 = ConvTranspose2d(features*4, features*2, 2, 2)
        self.conv15 = Conv2d(features*4, features*2, 3, 1, 1)
        self.bn15 = BatchNorm2d(features*2)
        self.conv16 = Conv2d(features*2, features*2, 3, 1, 1)
        self.bn16 = BatchNorm2d(features*2)
        self.upconv4 = ConvTranspose2d(features*2, features, 2, 2)
        self.conv17 = Conv2d(features*2, features, 3, 1, 1)
        self.bn17 = BatchNorm2d(features)
        self.conv18 = Conv2d(features, features, 3, 1, 1)
        self.bn18 = BatchNorm2d(features)
        self.conv19 = Conv2d(features, out_channels, 3, 1, 1)

        self.pool = MaxPool2d(2, 2)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()


    def forward(self, x):
        # Encoder
        x1 = self.relu(self.bn1(self.conv1(x)))      # Conv Block 1
        x1 = self.relu(self.bn2(self.conv2(x1)))     # Conv Block 2
        x2 = self.pool(x1)                           # Pooling 1
        x2 = self.relu(self.bn3(self.conv3(x2)))     # Conv Block 3
        x2 = self.relu(self.bn4(self.conv4(x2)))     # Conv Block 4
        x3 = self.pool(x2)                           # Pooling 2
        x3 = self.relu(self.bn5(self.conv5(x3)))     # Conv Block 5
        x3 = self.relu(self.bn6(self.conv6(x3)))     # Conv Block 6
        x4 = self.pool(x3)                           # Pooling 3
        x4 = self.relu(self.bn7(self.conv7(x4)))     # Conv Block 7
        x4 = self.relu(self.bn8(self.conv8(x4)))     # Conv Block 8
        x5 = self.pool(x4)                           # Pooling 4
        x5 = self.relu(self.bn9(self.conv9(x5)))     # Conv Block 9
        x5 = self.relu(self.bn10(self.conv10(x5)))   # Conv Block 10

        # Decoder
        x = self.upconv1(x5)                         # Upconv 1
        x = torch.cat([x4, x], dim=1)                # Concat 1
        x = self.relu(self.bn11(self.conv11(x)))     # Conv Block 11
        x = self.relu(self.bn12(self.conv12(x)))     # Conv Block 12
        x = self.upconv2(x)                          # Upconv 2
        x = torch.cat([x3, x], dim=1)                # Concat 2
        x = self.relu(self.bn13(self.conv13(x)))     # Conv Block 13
        x = self.relu(self.bn14(self.conv14(x)))     # Conv Block 14
        x = self.upconv3(x)                          # Upconv 3
        x = torch.cat([x2, x], dim=1)                # Concat 3
        x = self.relu(self.bn15(self.conv15(x)))     # Conv Block 15
        x = self.relu(self.bn16(self.conv16(x)))     # Conv Block 16
        x = self.upconv4(x)                          # Upconv 4
        x = torch.cat([x1, x], dim=1)                # Concat 4
        x = self.relu(self.bn17(self.conv17(x)))     # Conv Block 17
        x = self.relu(self.bn18(self.conv18(x)))     # Conv Block 18
        x = self.conv19(x)                           # Conv Block 19
        x = self.sigmoid(x)                          # Sigmoid Activation

        return x
