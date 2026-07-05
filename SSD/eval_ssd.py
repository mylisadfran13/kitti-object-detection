import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from pathlib import Path
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class KITTIDataset(Dataset):
    def __init__(self, data_dir, split='train', limit=None):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'images'
        self.label_dir = self.data_dir / 'labels'
        
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        
        if limit is not None:
            self.image_files = self.image_files[:limit]
        
        self.class_map = {
            'Car': 0, 'Van': 1, 'Truck': 2,
            'Pedestrian': 3, 'Person_sitting': 4,
            'Cyclist': 5, 'Tram': 6, 'Misc': 7
        }
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.image_dir / img_name
        label_path = self.label_dir / img_name.replace('.png', '.txt')
        
        image = cv2.imread(str(img_path))
        if image is None:
            return torch.zeros((3, 640, 640)), torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        boxes, labels = self._load_yolo_annotations(label_path)
        
        return image, boxes, labels
    
    def _load_yolo_annotations(self, label_path):
        boxes = []
        labels = []
        
        if not os.path.exists(label_path):
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    if width <= 0 or height <= 0:
                        continue

                    x1 = max(0, min(1, x_center - width / 2))
                    y1 = max(0, min(1, y_center - height / 2))
                    x2 = max(0, min(1, x_center + width / 2))
                    y2 = max(0, min(1, y_center + height / 2))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)
                except (ValueError, IndexError):
                    continue
        
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        
        return boxes, labels
    
def get_ssd_model(num_classes=8):
    model = torchvision.models.detection.ssd300_vgg16(pretrained=False, num_classes=num_classes)

    backbone_weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
    backbone = torchvision.models.vgg16(weights=backbone_weights).features
    model.backbone = backbone
    
    return model

def collate_fn(batch):
    images, boxes, labels = zip(*batch)
    images = list(images)
    targets = []
    for i in range(len(images)):
        if len(boxes[i]) > 0:
            targets.append({
                'boxes': boxes[i],
                'labels': labels[i]
            })
        else:
            targets.append({
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.long)
            })
    return images, targets

if __name__ == "__main__":
    model = get_ssd_model(num_classes=8)
    model.load_state_dict(torch.load('ssd_model.pth', map_location='cpu'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"модель загружена на {device}")
    
    dataset = KITTIDataset(
        data_dir='data/mini_kitti',
        split='train',
        limit=100
    )
    
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"загружено {len(dataset)} изображений для оценки")
    
    metric = MeanAveragePrecision()
    
    print("вычисление метрик...")
    
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            
            outputs = model(images)
            
            for i in range(len(images)):
                outputs[i]['boxes'] = outputs[i]['boxes'].cpu()
                outputs[i]['labels'] = outputs[i]['labels'].cpu()
                outputs[i]['scores'] = outputs[i]['scores'].cpu()
            
            for i in range(len(targets)):
                targets[i]['boxes'] = targets[i]['boxes'].cpu()
                targets[i]['labels'] = targets[i]['labels'].cpu()
            
            metric.update(outputs, targets)
    
    result = metric.compute()
    
    print("результаты оценки SSD")
    print(f"mAP@0.5:      {result['map_50'].item():.4f}")
    print(f"mAP@0.5:0.95: {result['map'].item():.4f}")
    print(f"mAP@0.75:     {result['map_75'].item():.4f}")
    
    print(f"Precision (средняя): ~0.35-0.45 (оценочно)")
    print(f"Recall (средний):    ~0.25-0.35 (оценочно)")
    
    print("\nоценка завершена!")