import torch
import SimpleITK as sitk
import numpy as np
import cv2
import os
from model import MultiTaskUNet
import config

class InferenceEngine:
    """
    推理引擎，处理DICOM图像并可视化分割结果
    """
    def __init__(self, model_path):
        self.cfg = config.Config()
        self.model = MultiTaskUNet(self.cfg.num_classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # 颜色映射
        self.colors = {
            0: (0, 0, 0),       # 背景 - 黑色
            1: (0, 255, 0),     # 类别1 - 绿色
            2: (0, 0, 255)      # 类别2 - 红色
        }
    
    def preprocess(self, image_path):
        """预处理DICOM图像"""
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image)
        
        # 标准化
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        
        # 转换为torch张量
        image_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)
        
        return image_tensor, image_array
    
    def predict(self, image_tensor):
        """预测分割结果"""
        with torch.no_grad():
            output = self.model(image_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0)
        return pred.numpy()
    
    def visualize(self, original, prediction, save_path=None):
        """可视化分割结果"""
        # 转换为8位灰度图像
        original = (original * 255).astype(np.uint8)
        
        # 创建彩色分割图
        seg_image = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        for class_id, color in self.colors.items():
            seg_image[prediction == class_id] = color
        
        # 叠加原始图像和分割结果
        overlay = cv2.addWeighted(
            cv2.cvtColor(original, cv2.COLOR_GRAY2BGR), 0.7,
            seg_image, 0.3, 0
        )
        
        # 显示或保存结果
        if save_path:
            cv2.imwrite(save_path, overlay)
        else:
            cv2.imshow('Segmentation Result', overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return overlay
    
    def process_dicom(self, dicom_path, output_path=None):
        """处理单个DICOM文件"""
        # 预处理
        image_tensor, original_image = self.preprocess(dicom_path)
        
        # 预测
        prediction = self.predict(image_tensor)
        
        # 可视化
        self.visualize(original_image[0], prediction[0], output_path)

if __name__ == "__main__":
    # 示例用法
    model_path = "models/unet_resnet.pth"
    dicom_path = "data/test/image.dcm"
    output_path = "results/segmentation_result.png"
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 运行推理
    engine = InferenceEngine(model_path)
    engine.process_dicom(dicom_path, output_path)
    print(f"Segmentation result saved to {output_path}")