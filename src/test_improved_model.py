import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from prepare_dataset import test_loader, snr_test
from improved_model import ImprovedModulationCNN
from train import evaluate


# =========================
# 1. 调制类型
# =========================

MODULATIONS = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM"
]


# =========================
# 2. 选择设备
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Testing device:", device)


# =========================
# 3. 创建改进后的模型
# =========================

model = ImprovedModulationCNN(
    num_classes=11,
    in_channels=2
)


# =========================
# 4. 加载最佳改进模型
# =========================

checkpoint_path = "checkpoints/best_model_improved.pt"

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)

model.to(device)


# =========================
# 5. 整体测试结果
# =========================

criterion = torch.nn.CrossEntropyLoss()

test_loss, test_acc = evaluate(
    model=model,
    loader=test_loader,
    criterion=criterion,
    device=device
)

print("\nTest result:")
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)
print("Test accuracy (%):", test_acc * 100)


# =========================
# 6. 收集所有预测结果
# =========================

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(device)

        logits = model(x)

        predictions = logits.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            y.numpy()
        )


all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)


# =========================
# 7. 按 SNR 计算准确率
# =========================

unique_snrs = np.sort(
    np.unique(snr_test)
)

snr_values = []
accuracy_values = []

print("\nAccuracy by SNR:")

for snr in unique_snrs:

    mask = snr_test == snr

    snr_predictions = all_predictions[mask]
    snr_labels = all_labels[mask]

    snr_accuracy = np.mean(
        snr_predictions == snr_labels
    )

    snr_values.append(snr)

    accuracy_values.append(
        snr_accuracy * 100
    )

    print(
        f"SNR {snr:>3} dB: "
        f"{snr_accuracy * 100:.2f}%"
    )


# =========================
# 8. 绘制 Accuracy vs SNR
# =========================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    snr_values,
    accuracy_values,
    marker="o"
)

plt.xlabel(
    "SNR (dB)"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Improved CNN Accuracy vs SNR"
)

plt.grid(True)

plt.ylim(
    0,
    100
)

plt.savefig(
    "results/improved_accuracy_vs_snr.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "\nFigure saved to: "
    "results/improved_accuracy_vs_snr.png"
)

plt.show()


# =========================
# 9. 全 SNR 混淆矩阵
# =========================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

cm_normalized = (
    cm.astype("float")
    / cm.sum(axis=1)[:, np.newaxis]
)

plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    cm_normalized,
    interpolation="nearest"
)

plt.title(
    "Improved CNN - Normalized Confusion Matrix - All SNR"
)

plt.colorbar()

tick_marks = np.arange(
    len(MODULATIONS)
)

plt.xticks(
    tick_marks,
    MODULATIONS,
    rotation=45
)

plt.yticks(
    tick_marks,
    MODULATIONS
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

for i in range(
    cm_normalized.shape[0]
):

    for j in range(
        cm_normalized.shape[1]
    ):

        plt.text(
            j,
            i,
            f"{cm_normalized[i, j]:.2f}",
            horizontalalignment="center",
            verticalalignment="center"
        )


plt.tight_layout()

plt.savefig(
    "results/improved_confusion_matrix_all_snr.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "Figure saved to: "
    "results/improved_confusion_matrix_all_snr.png"
)

plt.show()


# =========================
# 10. 只保留 SNR >= 0 dB
# =========================

high_snr_mask = snr_test >= 0

high_snr_predictions = (
    all_predictions[high_snr_mask]
)

high_snr_labels = (
    all_labels[high_snr_mask]
)

print(
    "\nSamples with SNR >= 0 dB:",
    len(high_snr_labels)
)

high_snr_acc = np.mean(
    high_snr_predictions
    == high_snr_labels
)

print(
    "Accuracy for SNR >= 0 dB:",
    high_snr_acc
)

print(
    "Accuracy for SNR >= 0 dB (%):",
    high_snr_acc * 100
)


# =========================
# 11. SNR >= 0 dB 混淆矩阵
# =========================

cm_high = confusion_matrix(
    high_snr_labels,
    high_snr_predictions
)

cm_high_normalized = (
    cm_high.astype("float")
    / cm_high.sum(axis=1)[:, np.newaxis]
)

plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    cm_high_normalized,
    interpolation="nearest"
)

plt.title(
    "Improved CNN - Normalized Confusion Matrix - SNR >= 0 dB"
)

plt.colorbar()

plt.xticks(
    tick_marks,
    MODULATIONS,
    rotation=45
)

plt.yticks(
    tick_marks,
    MODULATIONS
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

for i in range(
    cm_high_normalized.shape[0]
):

    for j in range(
        cm_high_normalized.shape[1]
    ):

        plt.text(
            j,
            i,
            f"{cm_high_normalized[i, j]:.2f}",
            horizontalalignment="center",
            verticalalignment="center"
        )


plt.tight_layout()

plt.savefig(
    "results/improved_confusion_matrix_snr_ge_0.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "Figure saved to: "
    "results/improved_confusion_matrix_snr_ge_0.png"
)

plt.show()