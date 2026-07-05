import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

def run_experiment(lr, batch, name):
    print(f"\nэксперимент: {name} (lr={lr}, batch={batch})")
    model = YOLO('yolov8n.pt')
    model.train(
        data='configs/mini_kitti.yaml',
        epochs=10,
        imgsz=640,
        batch=batch,
        lr0=lr,
        device=0,
        project='results/hyper_yolo',
        name=name,
        exist_ok=True,
        workers=4 
    )
    print(f"{name} завершен!")

if __name__ == '__main__':
    print("поиск гиперпараметров для YOLO")
    run_experiment(lr=0.01, batch=8, name='exp1_baseline')
    
    run_experiment(lr=0.005, batch=8, name='exp2_lr005')
    
    run_experiment(lr=0.01, batch=16, name='exp3_batch16')
    
    print("все эксперименты завершены!")