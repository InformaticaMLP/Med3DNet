import torch
import SimpleITK as sitk
import numpy as np
import random
from torch.utils.data import Dataset

class MedicalImageDataset(Dataset):
    """
    医学图像数据集类，支持DICOM格式和3D数据增强
    """
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取DICOM图像
        image = sitk.ReadImage(self.image_paths[idx])
        image_array = sitk.GetArrayFromImage(image)
        
        # 读取标签
        label = sitk.ReadImage(self.label_paths[idx])
        label_array = sitk.GetArrayFromImage(label)
        
        # 数据预处理
        image_array = self.normalize(image_array)
        
        # 3D数据增强
        if self.transform:
            image_array, label_array = self.transform(image_array, label_array)
            
        # 转换为torch张量
        image_tensor = torch.FloatTensor(image_array).unsqueeze(0)  # 添加通道维度
        label_tensor = torch.LongTensor(label_array)
        
        return image_tensor, label_tensor
    
    def normalize(self, image):
        """标准化图像数据"""
        return (image - image.min()) / (image.max() - image.min())


def random_3d_rotation(image, label):
    """随机3D旋转增强"""
    axes = random.sample([0, 1, 2], k=3)
    angle = random.uniform(-15, 15)
    
    # 对图像和标签应用相同的旋转
    image = rotate(image, angle=angle, axes=axes, reshape=False)
    label = rotate(label, angle=angle, axes=axes, reshape=False)
    
    return image, label

# 其他3D增强函数可以在这里添加...