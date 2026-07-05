import os
from pathlib import Path
from PIL import Image

KITTI_ROOT = "data/mini_kitti"
IMAGES_DIR = os.path.join(KITTI_ROOT, "images")
LABELS_DIR = os.path.join(KITTI_ROOT, "labels")
YOLO_LABELS_DIR = os.path.join(KITTI_ROOT, "labels_yolo")

CLASS_MAP = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7,
}

os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

converted = 0
skipped = 0

for label_file in sorted(Path(LABELS_DIR).glob("*.txt")):
    image_path = os.path.join(IMAGES_DIR, label_file.stem + ".png")
    
    if not os.path.exists(image_path):
        image_path = os.path.join(IMAGES_DIR, label_file.stem + ".jpg")
        if not os.path.exists(image_path):
            print(f"нет изображения для {label_file.name}, пропуск")
            skipped += 1
            continue
    
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    
    yolo_lines = []
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            
            cls_name = parts[0]
            if cls_name == "DontCare" or cls_name not in CLASS_MAP:
                continue
            
            cls_id = CLASS_MAP[cls_name]

            left, top, right, bottom = map(float, parts[4:8])
            
            x_center = ((left + right) / 2) / img_w
            y_center = ((top + bottom) / 2) / img_h
            width = (right - left) / img_w
            height = (bottom - top) / img_h
            
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            width = min(max(width, 0), 1)
            height = min(max(height, 0), 1)
            
            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    out_path = os.path.join(YOLO_LABELS_DIR, label_file.name)
    with open(out_path, "w") as f:
        f.write("\n".join(yolo_lines))
    
    converted += 1

print(f"конвертировано файлов: {converted}")
print(f"пропущено: {skipped}")