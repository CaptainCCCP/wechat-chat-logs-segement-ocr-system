"""
get semantic or instance segmentation annotations from coco data set.
if semantic: use line35、38
if instance: use line36、39
"""
from PIL import Image
import imgviz
import argparse
import os
import tqdm
import shutil
import numpy as np
from pycocotools.coco import COCO
 
def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)
 
def main(args):
    annotation_file = "C:/Users/moige/Desktop/ocr/maskrcnn-test/data_dataset_coco/annotations.json"# os.path.join(args.input_dir, 'annotations.json')
    # os.makedirs(os.path.join(args.input_dir, 'data_dataset_coco'), exist_ok=True)
    # os.makedirs(os.path.join(args.input_dir, 'JPEGImages'), exist_ok=True)
    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    empty_anno = []
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(annIds) > 0:
            # mask = coco.annToMask(anns[0]) * anns[0]['category_id']
            mask = coco.annToMask(anns[0])
            for i in range(len(anns) - 1):
                # mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
                mask += coco.annToMask(anns[i + 1]) * (i + 2)
            img_origin_path = os.path.join(args.input_dir, args.split, img['file_name'])
            img_output_path = os.path.join(args.input_dir, img['file_name'])
            seg_output_path = os.path.join(args.input_dir, 'output',
                                           img['file_name'].replace('.jpg', '.png'))
            # shutil.copy(img_origin_path, img_output_path)
            save_colored_mask(mask, seg_output_path)
        else:
            empty_anno.append(imgId)
    print("No annotations images:", empty_anno)
    print("The number is ", len(empty_anno))
 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="C:/Users/moige/Desktop/ocr/maskrcnn-test/data_dataset_coco", type=str,
                        help="input dataset directory")
    parser.add_argument("--split", default="", type=str,
                        help="train2017 or val2017")
    return parser.parse_args()
 
if __name__ == '__main__':
    args = get_args()
    main(args)
    # 产生了单独标注的图像在output