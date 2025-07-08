import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class AttentionBlock(nn.Module):
    """注意力机制模块"""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention

class ResidualBlock(nn.Module):
    """残差块，用于替换标准卷积"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class MultiTaskUNet(nn.Module):
    """3D U-Net与ResNet结合的多任务模型"""
    def __init__(self, num_classes):
        super(MultiTaskUNet, self).__init__()
        
        # 编码器部分（下采样）
        self.encoder1 = ResidualBlock(1, 64)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool3d(2)
        self.encoder4 = ResidualBlock(256, 512)
        
        # 注意力机制
        self.attention = AttentionBlock(512)
        
        # 解码器部分（上采样）
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock(512, 256)
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(256, 128)
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(128, 64)
        
        # 多任务输出
        self.segmentation = nn.Conv3d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # 注意力
        enc4 = self.attention(enc4)
        
        # 解码器
        dec3 = self.upconv3(enc4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # 输出
        seg_output = self.segmentation(dec1)
        
        return seg_output