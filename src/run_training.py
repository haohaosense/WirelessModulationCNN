import torch

from prepare_dataset import train_loader, val_loader
from model import ModulationCNN
from train import train


# =========================
# 1. 选择训练设备
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\nTraining device:", device)


# =========================
# 2. 创建 CNN 模型
# =========================

model = ModulationCNN(
    num_classes=11,
    in_channels=2
)


# =========================
# 3. 开始训练
# =========================

history = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    lr=1e-3,
    device=device
)