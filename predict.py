import os
import numpy as np
import torch
from PIL import Image

import torchvision

import sys
sys.path.append("./detection")
import detection.transforms as T
import cv2

import random
import time
import datetime

def random_color():
    b = random.randint(0,255)
    g = random.randint(0,255)
    r = random.randint(0,255)
 
    return (b,g,r)


def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1))) # 转换为tensor
    return img.float().div(255)  # 255也可以改为256  # 化为全部0-1的

def PredictImg(image, model,device):
    #img, _ = dataset_test[0] 
    img = cv2.imread(image)  # 读取img
    result = img.copy()
    dst=img.copy()  # 复制两个result和dst用于绘制
    img=toTensor(img)  # 转为imgtensor

    names = {'0': 'background', '1': 'user'}
    # put the model in evaluati
    # on mode

    prediction = model([img.to(device)])  # 预测结果

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    masks=prediction[0]['masks']

    m_bOK=False
    region_count = 0
    regions = []
    for idx in range(boxes.shape[0]):
        if scores[idx] >= 0.3:
            m_bOK=True
            color=random_color()
            mask=masks[idx, 0].mul(255).byte().cpu().numpy()
            thresh = mask  # 大于标准的掩码
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(dst, contours, -1, color, -1)

            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = names.get(str(labels[idx].item()))
            cv2.rectangle(result,((int(x1),int(y1))),((int(x2),int(y2))),color,thickness=2)
            cv2.putText(result, text=name, org=(int(x1), int(y1+10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=color)

            dst1=cv2.addWeighted(result,0.7,dst,0.5,0)  # 图像融合，result是绘制过的图
            
            # Crop the region with label and append it to the list
            region = dst1[int(y1):int(y2), int(x1):int(x2)]
            regions.append((region, y1))  # Store the region along with its y-coordinate
            
    if m_bOK:
        regions.sort(key=lambda r: r[1])  # Sort regions by y-coordinate (top to bottom)
        

    # Save the regions as separate images with sorted IDs
        for i, (region, _) in enumerate(regions):
            cv2.imwrite(f'maskrcnn-test/generated/region_{i}.jpg', region)
        cv2.namedWindow('result', 0)
        # cv2.resizeWindow("result",1200,700)
        cv2.imshow('result',dst1)
        cv2.waitKey()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    model.to(device)
    model.eval()
    save = torch.load("maskrcnn-test\FQmodel.pth")
    model.load_state_dict(save)
    start_time = time.time()
    PredictImg('maskrcnn-test/testimg.jpg',model,device)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print(total_time)
