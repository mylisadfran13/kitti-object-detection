import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, 11)
loss_values = [4.5, 3.8, 3.2, 2.9, 2.6, 2.4, 2.2, 2.0, 1.9, 1.8]
map_values = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.13, 0.14, 0.15, 0.16]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss_values, marker='o', linewidth=2, color='#6A994E')
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Функция потерь', fontsize=12)
plt.title('EfficientDet: Функция потерь', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(epochs)
plt.ylim(0, 5.5)

plt.subplot(1, 2, 2)
plt.plot(epochs, map_values, marker='s', linewidth=2, color='#F18F01')
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('mAP@0.5', fontsize=12)
plt.title('EfficientDet: Рост точности', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(epochs)
plt.ylim(0, 0.25)

plt.tight_layout()
plt.savefig('efficientdet_curves_real.png', dpi=150)
plt.show()
print("График сохранён как efficientdet_curves_real.png")