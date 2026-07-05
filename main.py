import os
import torch
from torch.utils.data import DataLoader
from src.dataset.kitti_dataset import KITTIDataset

def test_dataset():
    print("тестирование загрузки датасета...")
    
    train_dataset = KITTIDataset(
        data_dir='data/raw',
        split='train',
        limit=10  
    )
    
    print(f"загружено изображений: {len(train_dataset)}")
    
    if len(train_dataset) > 0:
        image, boxes, labels = train_dataset[0]
        print(f"размер изображения: {image.shape}")
        print(f"количество объектов: {len(boxes)}")
        print(f"метки: {labels}")
    else:
        print("нет данных!")
    
    return True

if __name__ == "__main__":
    test_dataset()