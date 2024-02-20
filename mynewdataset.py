import os
import torch
import numpy as np
import torch.utils.data
from PIL import Image
import detection.utils
import matplotlib.pyplot as plt
# ===========================================================================================================================
# 定义dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root # 数据集的根路径
        self.transforms = transforms # 数据集的预处理变形参数
        
        # 路径组合后返回该路径下的排序过的文件名（排序是为了对齐）
        # list(sorted(os.listdir(os.path.join(root, "output/JPEGImages"))))
        self.imgs = sorted(os.listdir(os.path.join(root, "JPEGImages")), key=lambda x: int(x.split('.')[0])) # self.imgs 是一个全部待训练图片文件名的有序列表
        # self.masks = sorted(os.listdir(os.path.join(root, "output/JPEGImages")), key=lambda x: int(x.split('.')[0])) # self.masks 是一个全部掩码图片文件名的有序列表
        self.masks = [file for file in sorted(os.listdir(os.path.join(root, "output/JPEGImages")), key=lambda x: int(x.split('.')[0])) if file.endswith(".png")]
        # print(self.imgs)
        # print(self.masks)
 
    # 根据idx对应读取待训练图片以及掩码图片
    def __getitem__(self, idx):
        # 根据idx针对img与mask组合路径
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "output/JPEGImages", self.masks[idx])
        
        # 根据路径读取三色图片并转为RGB格式
        img = Image.open(img_path).convert("RGB")
        
        # 根据路径读取掩码图片默认“L”格式
        mask = Image.open(mask_path)
        # 将mask转为numpy格式，h*w的矩阵,每个元素是一个颜色id
        mask = np.array(mask)
        # print(mask)
        # ==================================================================================
        # 获取mask中的id组成列表，obj_ids=[0,1,2]
        obj_ids = np.unique(mask)
        # 列表中第一个元素代表背景，不属于我们的目标检测范围，obj_ids=[1,2]
        obj_ids = obj_ids[1:]
 
        # obj_ids[:,None,None]:[[[1]],[[2]]],masks(2,536,559)
        # 为每一种类别序号都生成一个布尔矩阵，标注每个元素是否属于该颜色
        masks = mask == obj_ids[:, None, None]
 
        # 为每个目标计算边界框，存入boxes
        num_objs = len(obj_ids) # 目标个数N
        boxes = [] # 边界框四个坐标的列表，维度(N,4)
        for i in range(num_objs):
            pos = np.where(masks[i]) # pos为mask[i]值为True的地方,也就是属于该颜色类别的id组成的列表
            xmin = np.min(pos[1]) # pos[1]为x坐标，x坐标的最小值
            xmax = np.max(pos[1])
            ymin = np.min(pos[0]) # pos[0]为y坐标
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
 
        # 将boxes转化为tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # 初始化类别标签
        labels = torch.ones((num_objs,), dtype=torch.int64) # labels[1,1] (2,)的array
        masks = torch.as_tensor(masks, dtype=torch.uint8) # 将masks转换为tensor
 
        # 将图片序号idx转换为tensor
        image_id = torch.tensor([idx])
        # 计算每个边界框的面积
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 假设所有实例都不是人群
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # iscrowd[0,0] (2,)的array
 
        # 生成一个字典
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        # 变形transform
        if self.transforms is not None:
            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)
    
# ===========================================================================
# 定义训练集和测试集类与dataloader
# get_transform分别为训练集和测试集获得transform参数

dataset = MyDataset('maskrcnn-test/data_dataset_coco')
# TODO
dataset_test = MyDataset('maskrcnn-test/data_dataset_coco')
 
torch.manual_seed(1)
# 返回一个包含数据集标号的随机列表，相当于随机化打乱标号
# torch.randperm(4).tolist() [2,1,0,3]

indices = torch.randperm(len(dataset)).tolist()
# 完成训练集和测试集的分割，dataset取dataset第一个到第倒数80个，dataset_test取倒数80
dataset = torch.utils.data.Subset(dataset, indices[:-4])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-4:])
 
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=False, num_workers=0,
    collate_fn=detection.utils.collate_fn
    ) # collate_fn是取样本的方法参数
 
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=detection.utils.collate_fn)

# ====================================================================
# 查看data_loader输出
# print(data_loader.dataset.__getitem__(0))

# for data in data_loader:
#     # print(data)
#     index = data[0]  # 第一个元素是图像数据
#     target = data[1]  # 第二个元素是目标数据

#     print(data[0][0])
#     exit()
    # plt.imshow(target[0][0])
    # plt.axis('off')
    # plt.show()

    # plt.imshow(target[0][1])
    # plt.axis('off')
    # plt.show()

# target[0][0]target[0][1]是图片,有Image object,image mode,size;
# target[1][0]target[1][1]是对应信息，有boxes, labels,masks,image_id,area,iscrowd
# 数据是这样的
    
# ((<PIL.Image.Image image mode=RGB size=1170x2532 at 0x176F3080A60>, <PIL.Image.Image image mode=RGB size=1170x2532 at 0x176F3124BB0>), 
#  ({'boxes': tensor([]), 
#    'labels': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1]), 
#    'masks': tensor([], dtype=torch.uint8), 
#    'image_id': tensor([13]), 
#    'area': tensor([46612., 42054., 55380., 83258., 44763., 60434., 55948., 39767., 53466.]), 
#    'iscrowd': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])},
    
#  {'boxes': tensor([]), 
#   'labels': tensor([1, 1, 1]), 
#   'masks': tensor([], dtype=torch.uint8), 
#   'image_id': tensor([2]), 
#   'area': tensor([671370., 886414.,  47955.]), 
#   'iscrowd': tensor([0, 0, 0])}))