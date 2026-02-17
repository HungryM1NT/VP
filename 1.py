import numpy as np
import matplotlib.pyplot as plt


def generate_empirical_tp(mu, m_samples=6000):
    # 1. Генерируем выборку
    samples = -(1/mu) * np.log(np.random.random(m_samples))
    # 2. Строим гистограмму (24 интервала)
    hist, bin_edges = np.histogram(samples, bins=24, range=(0, 10), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Нормируем вероятности для отрезков
    probs = hist / np.sum(hist)
    cum_probs = np.cumsum(probs)
    return cum_probs, bin_centers

mu1, mu2 = 1.0, 0.5
m = 60 # число реализаций суммы

cum_p1, centers1 = generate_empirical_tp(mu1)
cum_p2, centers2 = generate_empirical_tp(mu2)

tp_sum_list = []

for _ in range(m):
    # Генерируем случайное число r для каждого процесса
    r1, r2 = np.random.random(), np.random.random()
    
    # Находим интервал (индекс), в который попало число
    idx1 = np.searchsorted(cum_p1, r1)
    idx2 = np.searchsorted(cum_p2, r2)
    
    # Ограничиваем индекс длиной массива
    idx1 = min(idx1, len(centers1)-1)
    idx2 = min(idx2, len(centers2)-1)
    
    tp1 = centers1[idx1]
    tp2 = centers2[idx2]
    
    tp_sum_list.append(tp1 + tp2)

# Визуализация суммы процессов
plt.figure(figsize=(8, 5))
plt.hist(tp_sum_list, bins=15, color='purple', alpha=0.7, density=True)
plt.title("Гистограмма суммы процессов $Tp_{sum} = Tp_1 + Tp_2$")
plt.xlabel("Время (мс)")
plt.ylabel("Плотность вероятности")
plt.grid(axis='y', linestyle='--')
plt.show()