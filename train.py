import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse
import os
from model import MultiTaskUNet
from data_preprocessing import MedicalImageDataset
import config

# 损失函数
class CombinedLoss(nn.Module):
    """Dice损失和交叉熵的联合损失"""
    def __init__(self, weight_ce=0.5, weight_dice=0.5):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss()
        
    def dice_loss(self, pred, target):
        smooth = 1.
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        
        return 1 - ((2. * intersection + smooth) / 
                   (iflat.sum() + tflat.sum() + smooth))
    
    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice_loss(F.softmax(pred, dim=1), target)
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss

# 训练函数
def train_model():
    # 初始化分布式训练
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    # 加载配置
    cfg = config.Config()
    
    # 数据集和数据加载器
    train_dataset = MedicalImageDataset(cfg.train_images, cfg.train_labels)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, 
                             sampler=train_sampler, num_workers=4)
    
    # 模型
    model = MultiTaskUNet(cfg.num_classes).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = CombinedLoss()
    
    # 训练循环
    for epoch in range(cfg.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(local_rank), target.to(local_rank)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % cfg.log_interval == 0 and local_rank == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}')
    
    # 保存模型
    if local_rank == 0:
        torch.save(model.module.state_dict(), cfg.model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    train_model()