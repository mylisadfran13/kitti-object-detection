import matplotlib.pyplot as plt

epochs = list(range(1, 11))

loss_exp1 = [1.70, 1.56, 1.48, 1.46, 1.40, 1.37, 1.34, 1.32, 1.27, 1.28]
loss_exp2 = [1.70, 1.56, 1.48, 1.46, 1.40, 1.37, 1.34, 1.32, 1.27, 1.28]  
loss_exp3 = [1.69, 1.59, 1.48, 1.47, 1.42, 1.40, 1.32, 1.29, 1.27, 1.26]

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_exp1, marker='o', label='базовый (lr=0.01, batch=8)', color='#2E86AB')
plt.plot(epochs, loss_exp2, marker='s', label='меньший lr (lr=0.005, batch=8)', color='#F18F01')
plt.plot(epochs, loss_exp3, marker='^', label='больший batch (lr=0.01, batch=16)', color='#6A994E')

plt.xlabel('эпоха', fontsize=12)
plt.ylabel('функция потерь (box_loss)', fontsize=12)
plt.title('сравнение гиперпараметров YOLO', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(epochs)
plt.savefig('yolo_hyperparams_comparison.png', dpi=150)
plt.show()