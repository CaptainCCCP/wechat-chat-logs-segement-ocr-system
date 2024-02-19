# https://blog.csdn.net/m0_51325463/article/details/127516052?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170825791516800213084906%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=170825791516800213084906&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-127516052-null-null.142^v99^pc_search_result_base7&utm_term=%E7%BA%AF%E5%B0%8F%E7%99%BD%20%E5%8A%A8%E6%89%8B%E5%AE%9E%E7%8E%B0mask%20rcnn&spm=1018.2226.3001.4187
import os
import torch
import numpy as np
import torch.utils.data
from PIL import Image

from mydataset import data_loader, data_loader_test

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


from torchvision import transforms as T
from detection.engine import train_one_epoch, evaluate
# ===========================================================================
# 获取模型 
def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
 
    # 预训练模型的打分模块的输入维度，也就是特征提取模块的输出维度
    in_features = model.roi_heads.box_predictor.cls_score.in_features
 
    # 将预训练模型的预测部分修改为FastR-CNN的预测部分（Fast R-CNN与Faster R-CNN的预测部分相同）
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
 
    # 预训练模型中像素级别预测器的输入维度
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
 
    num_classes=2

    # 使用自己的参数生成Mask预测器替换预训练模型中的Mask预测器部分
    # 三个参数，输入维度，中间层维度，输出维度（类别个数）
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
 
    return model
# ===========================================================================
# TODO
# 首先将PIL图片转变为tensor格式
# 如果是训练集，那么加入随机水平翻转，否则不进行任何操作
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# ===========================================================================
# device=torch.device("cpu")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 目标分类个数user和chat
num_classes = 2
model = get_instance_segmentation_model(num_classes)
model.to(device)
 
# 定义待训练的参数以及优化器
params = [p for p in model.parameters() if p.requires_grad] # 模型中的参数，凡是需要计算梯度的，统统列为待训练参数
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
 
# 学习率的动态调整，每3个epochs将学习率缩小0.1倍
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
# =======================================================================================
# 训练
need_training = True
num_epochs = 10
if need_training == True:
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10,)
        # 手动更新学习率
        lr_scheduler.step()
        # 在测试集上评估模型
        #evaluate(model, data_loader_test, device=device)
# 保存模型
PATH = "maskrcnn-test\FQmodel.pth"
torch.save(model.state_dict(), PATH)

