import matplotlib.pyplot as plt

epochs = list(range(1, 21))
losses = [
    32.2050, 8.8842, 7.1420, 7.1858, 6.3555,
    6.9513, 4.9474, 4.4026, 4.1648, 4.3073,
    3.4577, 4.0988, 4.3104, 3.1476, 3.8373,
    4.2909, 3.2978, 2.7132, 2.8162, 2.3589
]

plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', linewidth=2, color='#2E86AB')
plt.xlabel('эпоха', fontsize=12)
plt.ylabel('функция потерь', fontsize=12)
plt.title('график изменения функции потерь при обучении SSD', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(epochs)
plt.ylim(0, 35)

plt.savefig('ssd_loss_curve.png', dpi=150, bbox_inches='tight')
plt.show()