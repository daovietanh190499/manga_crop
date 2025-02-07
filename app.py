from datetime import datetime
import torch
import cv2
from PIL import Image
from io import BytesIO
import base64
import traceback
import numpy as np
import os
import glob
import shutil
from textblockdetector import dispatch as dispatch_textdetector
import json

use_cuda = torch.cuda.is_available()

print(use_cuda)

import cv2
import numpy as np

with open("config.json") as f:
    config = json.load(f)

print(config)

def infer(img1, img2, img3, height, index, use_mask, not_effect=True):
    if img2 is None and img3 is None:
        img = img1
    elif img3 is None:
        img = np.vstack((img1, img2))
    else:
        img = np.vstack((img1, img2, img3))
    mask, mask_refined, blk_list = dispatch_textdetector(img, use_cuda)
    torch.cuda.empty_cache()

    mask = cv2.dilate((mask > 150).astype('uint8')*255, np.ones((7,7), np.uint8), iterations=5)
    kernel = np.ones((9,9), np.uint8)
    mask_refined = cv2.dilate(mask_refined, kernel, iterations=5)

    new_mask = np.zeros(img.shape[:2])


    if use_mask:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

        bboxes = []
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            bboxes.append([x, y, w, h])

    else:
        area = []
        fonts = []
        for i, blk in enumerate(blk_list):
            xmin, ymin, xmax, ymax = blk.xyxy
            xmin = 0 if xmin < 0 else xmin
            ymin = 0 if ymin < 0 else ymin
            xmax = img.shape[1] if xmax >  img.shape[1] else xmax
            ymax = img.shape[0] if ymax >  img.shape[0] else ymax
            area.append((xmax-xmin)*(ymax-ymin))
            fonts.append(blk.font_size)

        indexes = np.lexsort((-np.array(area), np.array(fonts)))

        # indexes = np.argsort(np.array(fonts).astype("float32"))#[::-1]
        new_blk_list = [blk_list[i] for i in indexes.tolist()]
        blk_list = new_blk_list

        bboxes = []
        new_blk_list = []
        filter_mask = np.zeros_like(mask)
        for i, blk in enumerate(blk_list):
            xmin, ymin, xmax, ymax = blk.xyxy
            xmin = 0 if xmin < 0 else xmin
            ymin = 0 if ymin < 0 else ymin
            xmax = img.shape[1] if xmax >  img.shape[1] else xmax
            ymax = img.shape[0] if ymax >  img.shape[0] else ymax
            fill_area = np.sum(new_mask[int(ymin):int(ymax), int(xmin):int(xmax)])
            if fill_area/((xmax-xmin)*(ymax-ymin)) <= 0.2:
                new_mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
                bboxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                new_blk_list.append(blk)

    final_text = []
    final_bboxes = None
    for i, bbox in enumerate(bboxes):
        xmin, ymin, w, h = bbox
        final_bboxes = np.concatenate((final_bboxes, np.array([[xmin, ymin, w, h]]))) if not final_bboxes is None else np.array([[xmin, ymin, w, h]])

    coords = []
    coords_head = []
    coords_height = []
    final_bboxes = np.array([]) if final_bboxes is None else final_bboxes

    for bbox in final_bboxes:
        xmin, ymin, w, h = bbox
        # Calculate xmax, ymax
        xmax = xmin + w
        ymax = ymin + h
        
    #     # Draw rectangle
    #     cv2.rectangle(img, 
    #                  (int(xmin), int(ymin)), 
    #                  (int(xmax), int(ymax)),
    #                  (0, 255, 0),  # Color in BGR (green)
    #                  2)  # Thickness
        
        coords.append(ymin)
        coords_head.append(1)
        coords_height.append(h)
        # coords.append(ymax)
        # coords_head.append(0)

    min_coord = 0
    split = height
  
    if len(coords) > 0:
        coords = np.sort(coords)
        coords_arg = np.argsort(coords)

        coords_head = [coords_head[j] for j in coords_arg]
        coords_height = [coords_height[j] for j in coords_arg]
        
        if index == 0:
            min_coord = coords[0]
        else:
            min_coord = 0

        for i, coord in enumerate(coords):
            if coord - min_coord >= height:
                split = coord
                # if split + coords_height[i] < max_height_ratio*height:
                #     split += coords_height[i] + 60
                if split - coords[0] > max_height_ratio*height and i > 0:
                    split = coords[i - 1]
                    if split - coords[0] < min_height_ratio*height:
                        split = coords[i - 1] + coords_height[i - 1] + 60
                        if split - coords[0] < height:
                            split = height + coords[0]
                elif split - coords[0] > max_height_ratio*height:
                    split = height + coords[0]

                if coords_head[i] == 0:
                    split += 30
                else:
                    split -= 30

                break

    print(split + min_coord, min_coord, img.shape[0])

    img_part = img[min_coord:split + min_coord, :, :]

    cv2.imwrite(f"split/part_{str(index).zfill(6)}.png", img_part)

    final_img = img[split + min_coord:, :, :]
    
    if final_img.shape[0] > height:
        final_img, new_index = infer(final_img, None, None, height, index + 1, use_mask, True)
    else:
        new_index = index

    if split + min_coord >= img.shape[0]*0.99 and img2 is None and img3 is None and not not_effect:
        final_img = None

    return final_img, new_index

path = config['image_path']
prefix = config['image_prefix']
extension = config['image_extension']
height = config['image_height']
start = config['start_index']
min_height_ratio = config['min_height_ratio']
max_height_ratio = config['max_height_ratio']
use_mask = config.get('use_mask', True)

if not os.path.exists("split"):
  os.mkdir("split")
i = 0
img_src = None
new_index = -1
while (img_src is not None) or (i == 0):
    if i == 0:
        img_src = cv2.imread(path + f"{prefix}{i + start}.{extension}")
    if i + 2 > len(os.listdir(path)) - 1:
        img1 = None
        img2 = None
    else:
        img1 = cv2.imread(path + f"{prefix}{i + 1 + start}.{extension}")
        img2 = cv2.imread(path + f"{prefix}{i + 2 + start}.{extension}")
    img_src, new_index = infer(img_src, img1, img2, height, new_index + 1, use_mask, False)
    i += 2
