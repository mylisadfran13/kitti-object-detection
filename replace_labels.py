import shutil
import os

shutil.rmtree("data/mini_kitti/labels")

shutil.move("data/mini_kitti/labels_yolo", "data/mini_kitti/labels")

cache_path = "data/mini_kitti/labels.cache"
if os.path.exists(cache_path):
    os.remove(cache_path)

print("папка labels заменена на YOLO-аннотации")