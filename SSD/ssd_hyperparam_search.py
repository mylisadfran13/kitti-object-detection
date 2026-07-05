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

def get_ssd_model(num_classes=8):
    model = torchvision.models.detection.ssd300_vgg16(pretrained=False, num_classes=num_classes)
    backbone_weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
    backbone = torchvision.models.vgg16(weights=backbone_weights).features
    model.backbone = backbone
    return model

def train_ssd_with_params(lr, batch_size, name, epochs=10):
    print(f"\nэксперимент: {name} (lr={lr}, batch={batch_size})")
    
    dataset = KITTIDataset('data/mini_kitti', limit=500)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = get_ssd_model(num_classes=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(epochs):
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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), f'ssd_{name}.pth')
    print(f"модель сохранена как ssd_{name}.pth")
    
    return losses

if __name__ == '__main__':
    print("поиск гиперпараметров для SSD")
    
    losses1 = train_ssd_with_params(lr=1e-4, batch_size=8, name='baseline')
    
    losses2 = train_ssd_with_params(lr=5e-5, batch_size=8, name='lr5e5')
    
    losses3 = train_ssd_with_params(lr=1e-4, batch_size=16, name='batch16')
    
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, 11))
    
    plt.plot(epochs, losses1, marker='o', label='базовый (lr=1e-4, batch=8)', color='#2E86AB')
    plt.plot(epochs, losses2, marker='s', label='меньший lr (lr=5e-5, batch=8)', color='#F18F01')
    plt.plot(epochs, losses3, marker='^', label='больший batch (lr=1e-4, batch=16)', color='#6A994E')
    
    plt.xlabel('эпоха', fontsize=12)
    plt.ylabel('функция потерь', fontsize=12)
    plt.title('сравнение гиперпараметров SSD', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    plt.savefig('ssd_hyperparams_comparison.png', dpi=150)
    plt.show()
    
    print("все эксперименты SSD завершены!")