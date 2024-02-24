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
    assert type(img) == np.ndarray, 'The img type is {}, but ndarray is expected.'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1))) # Convert to tensor
    return img.float().div(255)  # Normalize to 0-1 range

def PredictImg(image, model, device):
    img = cv2.imread(image)
    result = img.copy()
    dst = img.copy()
    img = toTensor(img)

    names = {'0': 'background', '1': 'user'}

    prediction = model([img.to(device)])

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    masks = prediction[0]['masks']

    m_bOK = False
    regions = []

    for idx in range(boxes.shape[0]):
        if scores[idx] >= 0.32:
            m_bOK = True
            color = random_color()
            mask = masks[idx, 0].mul(255).byte().cpu().numpy()
            thresh = mask
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # cv2.drawContours(dst, contours, -1, color, -1)

            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = names.get(str(labels[idx].item()))
            # cv2.rectangle(result, ((int(x1), int(y1))), ((int(x2), int(y2))), color, thickness=2)
            # cv2.putText(result, text=name, org=(int(x1), int(y1 + 10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=color)

            dst1 = cv2.addWeighted(result, 0.7, dst, 0.5, 0)
            image_width = dst1.shape[1]

            x1 = max(0, x1 - 50)
            x2 = min(image_width, x2 + 50)
            region = dst1[int(y1):int(y2), int(x1):int(x2)]
            regions.append((region, (x1, y1, x2, y2)))

    if m_bOK:
        regions.sort(key=lambda r: (r[1][1], r[1][0]))  # Sort regions by y1, x1

        non_overlapping_regions = []
        non_overlapping_regions.append(regions[0])

        for i in range(1, len(regions)):
            current_region = regions[i][1]
            overlapping = False

            for j in range(len(non_overlapping_regions)):
                saved_region = non_overlapping_regions[j][1]

                if current_region[0] <= saved_region[2] and current_region[2] >= saved_region[0] and current_region[1] <= saved_region[3] and current_region[3] >= saved_region[1]:
                    overlapping = True
                    area_current = (current_region[2] - current_region[0]) * (current_region[3] - current_region[1])
                    area_saved = (saved_region[2] - saved_region[0]) * (saved_region[3] - saved_region[1])

                    if area_current > area_saved:
                        non_overlapping_regions[j] = regions[i]
                    break

            if not overlapping:
                non_overlapping_regions.append(regions[i])
        
        for i, (region, _) in enumerate(non_overlapping_regions):
            # if region is None or region.size == 0:
            #     continue
            cv2.imwrite(f'maskrcnn-test/generated/region_{i}.jpg', region)

    cv2.namedWindow('result', 0)
    cv2.imshow('result', dst1)
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
