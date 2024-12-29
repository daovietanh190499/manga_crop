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

use_cuda = torch.cuda.is_available()

import cv2
import numpy as np

with open("config.txt") as f:
    lines = f.readlines()

print(lines)

def infer(img1, img2, img3, height, index):
    img = np.vstack((img1, img2, img3))
    mask, mask_refined, blk_list = dispatch_textdetector(img, use_cuda)
    torch.cuda.empty_cache()

    mask = cv2.dilate((mask > 170).astype('uint8')*255, np.ones((5,5), np.uint8), iterations=5)
    kernel = np.ones((9,9), np.uint8)
    mask_refined = cv2.dilate(mask_refined, kernel, iterations=2)

    new_mask = np.zeros(img.shape[:2])

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

    blk_list = new_blk_list

    final_text = []
    final_bboxes = None
    for i, bbox in enumerate(bboxes):
        xmin, ymin, w, h = bbox
        final_bboxes = np.concatenate((final_bboxes, np.array([[xmin, ymin, w, h]]))) if not final_bboxes is None else np.array([[xmin, ymin, w, h]])

    coords = []
    coords_head = []
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
        coords.append(ymax)
        coords_head.append(0)

    min_coord = 0
    split = height
  
    if len(coords) > 0:
      coords = np.sort(coords)
      coords_arg = np.argsort(coords)

      coords_head = coords_head[coords_arg]
      
      if index == 0:
        min_coord = coords[0]

      coords -= min_coord

      coords_mask = coords > height

      for i, coord in enumerate(coords_mask):
        if coord:
          if not coords_head[i]:
            split = coords[i]
          else:
            split = int((coords[i] + height)/2)
          break

    print(split + min_coord, img.shape[0])

    img_part = img[:split + min_coord, :, :]

    cv2.imwrite(f"split/part_{str(index).zfill(6)}.png", img_part)

    return final_bboxes, img[split + min_coord:, :, :]

path = lines[0].strip()
prefix = lines[1].strip()
height = int(lines[2].strip())

if not os.path.exists("split"):
  os.mkdir("split")
i = 0
while i <= (len(os.listdir(path)) - 3):
  if i == 0:
    img_src = cv2.imread(path + f"{prefix}{i + 1}.jpg")
  img1 = cv2.imread(path + f"{prefix}{i + 2}.jpg")
  img2 = cv2.imread(path + f"{prefix}{i + 3}.jpg")
  final_bboxes, img_src = infer(img_src, img1, img2, height, i//2)
  i += 2