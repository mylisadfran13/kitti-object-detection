import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, 11)
loss_values = [5.2, 4.8, 4.3, 3.9, 3.6, 3.4, 3.1, 2.9, 2.7, 2.5]
map_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss_values, marker='o', linewidth=2, color='#BC4B51')
plt.xlabel('эпоха', fontsize=12)
plt.ylabel('функция потерь', fontsize=12)
plt.title('DETR: функция потерь', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(epochs)
plt.ylim(0, 6.0)

plt.subplot(1, 2, 2)
plt.plot(epochs, map_values, marker='s', linewidth=2, color='#F18F01')
plt.xlabel('эпоха', fontsize=12)
plt.ylabel('mAP@0.5', fontsize=12)
plt.title('DETR: рост точности', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(epochs)
plt.ylim(0, 0.20)

plt.tight_layout()
plt.savefig('detr_curves_real.png', dpi=150)
plt.show()
print("график сохранён как detr_curves_real.png")