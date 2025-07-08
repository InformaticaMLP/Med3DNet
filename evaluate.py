import torch
import numpy as np
import SimpleITK as sitk
from model import MultiTaskUNet
from data_preprocessing import MedicalImageDataset
import config
import os
import json
from datetime import datetime

class Evaluator:
    """
    评估类，计算Dice系数并生成报告
    """
    def __init__(self, model_path):
        self.cfg = config.Config()
        self.model = MultiTaskUNet(self.cfg.num_classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # 验证数据集
        self.val_dataset = MedicalImageDataset(
            self.cfg.val_images, 
            self.cfg.val_labels
        )
        
    def dice_coefficient(self, pred, target):
        """计算Dice系数"""
        smooth = 1.
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        
        return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    
    def evaluate(self):
        """评估模型性能"""
        dice_scores = []
        
        with torch.no_grad():
            for image, label in self.val_dataset:
                # 预测
                output = self.model(image.unsqueeze(0))
                pred = torch.argmax(output, dim=1).squeeze(0)
                
                # 计算Dice
                dice = self.dice_coefficient(pred, label)
                dice_scores.append(dice.item())
        
        # 统计结果
        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        
        # 生成报告
        report = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": os.path.basename(model_path),
            "num_samples": len(dice_scores),
            "mean_dice": mean_dice,
            "std_dice": std_dice,
            "dice_scores": dice_scores
        }
        
        # 保存报告
        report_path = os.path.join('reports', 'evaluation_report.json')
        os.makedirs('reports', exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        return report

if __name__ == "__main__":
    model_path = "models/unet_resnet.pth"
    evaluator = Evaluator(model_path)
    report = evaluator.evaluate()
    print(f"Mean Dice: {report['mean_dice']:.4f} ± {report['std_dice']:.4f}")