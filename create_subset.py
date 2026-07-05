import os
import shutil
from pathlib import Path

src_img = 'data/raw/training/image_2'
src_lbl = 'data/raw/training/label_2'
dst_img = 'data/mini_kitti/images'
dst_lbl = 'data/mini_kitti/labels'

os.makedirs(dst_img, exist_ok=True)
os.makedirs(dst_lbl, exist_ok=True)

image_files = sorted([f for f in os.listdir(src_img) if f.endswith('.png')])[:500]

print(f"найдено {len(image_files)} изображений")

copied = 0
for img_file in image_files:
    shutil.copy2(os.path.join(src_img, img_file), os.path.join(dst_img, img_file))
    
    txt_file = img_file.replace('.png', '.txt')
    src_txt = os.path.join(src_lbl, txt_file)
    if os.path.exists(src_txt):
        shutil.copy2(src_txt, os.path.join(dst_lbl, txt_file))
        copied += 1

print(f"скопировано {copied} изображений и аннотаций в data/mini_kitti/")