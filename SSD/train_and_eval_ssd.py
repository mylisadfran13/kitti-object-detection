import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt

class KITTIDataset(Dataset):
    def __init__(self, data_dir, limit=None):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'images'
        self.label_dir = self.data_dir / 'labels'
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        if limit:
            self.image_files = self.image_files[:limit]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.image_dir / img_name
        label_path = self.label_dir / img_name.replace('.png', '.txt')
        
        image = cv2.imread(str(img_path))
        if image is None:
            return torch.zeros((3, 640, 640)), [], []
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        boxes, labels = self._load_labels(label_path)
        return image, boxes, labels
    
    def _load_labels(self, label_path):
        boxes, labels = [], []
        if not os.path.exists(label_path):
            return [], []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    if w <= 0 or h <= 0:
                        continue
                    x1 = max(0, xc - w/2)
                    y1 = max(0, yc - h/2)
                    x2 = min(1, xc + w/2)
                    y2 = min(1, yc + h/2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)
                except:
                    continue
        return boxes, labels


def collate_fn(batch):
    images, boxes, labels = zip(*batch)
    images = list(images)
    targets = []
    for i in range(len(images)):
        if len(boxes[i]) > 0:
            targets.append({
                'boxes': torch.tensor(boxes[i], dtype=torch.float32),
                'labels': torch.tensor(labels[i], dtype=torch.long)
            })
        else:
            targets.append({
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.long)
            })
    return images, targets


def get_model(num_classes=8):
    model = torchvision.models.detection.ssd300_vgg16(pretrained=False, num_classes=num_classes)
    backbone = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
    model.backbone = backbone
    return model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"устройство: {device}")

    dataset = KITTIDataset('data/mini_kitti', limit=500)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = get_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses = []
    for epoch in range(10):
        model.train()
        total_loss = 0
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses_ = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses_.backward()
            optimizer.step()
            total_loss += losses_.item()
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'ssd_model_fixed.pth')
    print("М\модель сохранена")

    print("\nоценка модели...")
    model.eval()
    val_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    total_objects = 0
    detected = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(images)):
                total_objects += len(targets[i]['labels'])

                if len(outputs[i]['scores']) > 0:
                    high_conf = outputs[i]['scores'] > 0.5
                    detected += high_conf.sum().item()


    print(f"Ввего объектов в выборке: {total_objects}")
    print(f"найдено объектов (conf > 0.5): {detected}")
    if total_objects > 0:
        print(f"примерный Recall: {detected / total_objects:.4f}")
    else:
        print("объектов не найдено.")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), losses, marker='o', color='#2E86AB')
    plt.xlabel('эпоха')
    plt.ylabel('Loss')
    plt.title('SSD Training Loss')
    plt.grid(True)
    plt.savefig('ssd_loss_fixed.png', dpi=150)
    print("график сохранён как ssd_loss_fixed.png")