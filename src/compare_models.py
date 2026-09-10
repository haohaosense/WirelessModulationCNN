import matplotlib.pyplot as plt


# =========================
# 1. SNR
# =========================

snr_values = [
    -20, -18, -16, -14, -12,
    -10, -8, -6, -4, -2,
    0, 2, 4, 6, 8,
    10, 12, 14, 16, 18
]


# =========================
# 2. Baseline CNN 准确率
# =========================

baseline_accuracy = [
    9.52,
    9.52,
    9.27,
    12.24,
    16.36,
    25.33,
    36.00,
    50.48,
    64.36,
    74.48,
    79.94,
    80.61,
    82.12,
    83.27,
    82.97,
    82.61,
    83.45,
    83.58,
    82.91,
    83.52
]


# =========================
# 3. Improved CNN 准确率
# =========================

improved_accuracy = [
    9.76,
    9.03,
    9.45,
    12.18,
    16.79,
    25.82,
    38.67,
    54.48,
    67.76,
    78.12,
    83.39,
    84.36,
    83.82,
    85.21,
    85.15,
    84.97,
    86.42,
    85.15,
    83.94,
    85.09
]


# =========================
# 4. 绘图
# =========================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    snr_values,
    baseline_accuracy,
    marker="o",
    label="Baseline CNN"
)

plt.plot(
    snr_values,
    improved_accuracy,
    marker="s",
    label="Improved CNN"
)


# =========================
# 5. 图表信息
# =========================

plt.xlabel(
    "SNR (dB)"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Baseline vs Improved CNN"
)

plt.ylim(
    0,
    100
)

plt.grid(True)

plt.legend()


# =========================
# 6. 保存
# =========================

plt.savefig(
    "results/baseline_vs_improved.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "Figure saved to: "
    "results/baseline_vs_improved.png"
)


# =========================
# 7. 显示
# =========================

plt.show()