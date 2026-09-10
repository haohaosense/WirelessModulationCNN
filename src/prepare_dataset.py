import pickle
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


# =========================
# 1. 数据文件路径
# =========================

data_path = "data/RML2016.10a_dict.pkl"

print("Loading dataset...")

with open(data_path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

print("Dataset loaded.")


# =========================
# 2. 把原始字典整理成 X / y / snrs
# =========================

X = []
y = []
snrs = []

for key in data.keys():
    modulation, snr = key
    samples = data[key]

    for sample in samples:
        X.append(sample)
        y.append(modulation)
        snrs.append(snr)


# Python list → NumPy array

X = np.array(X)
y = np.array(y)
snrs = np.array(snrs)


print("\nFull dataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("snrs shape:", snrs.shape)


# =========================
# 3. 创建分层标签
# =========================
#
# 我们希望每一种：
#
# 调制类型 + SNR
#
# 在 Train / Validation / Test
# 中都保持大致相同的比例
#
# 例如：
# QPSK_2
# GFSK_6
# QAM16_-4
# ...


stratify_labels = np.array([
    f"{mod}_{snr}"
    for mod, snr in zip(y, snrs)
])


# =========================
# 4. 第一次划分
#
# 85% → 临时训练集
# 15% → Test
# =========================

(
    X_temp,
    X_test,
    y_temp,
    y_test,
    snr_temp,
    snr_test,
    strat_temp,
    strat_test
) = train_test_split(
    X,
    y,
    snrs,
    stratify_labels,
    test_size=0.15,
    random_state=42,
    stratify=stratify_labels
)


# =========================
# 5. 第二次划分
#
# 从剩下的 85% 中：
#
# 70% 总数据 → Train
# 15% 总数据 → Validation
#
# 0.17647 ≈ 15 / 85
# =========================

(
    X_train,
    X_val,
    y_train,
    y_val,
    snr_train,
    snr_val
) = train_test_split(
    X_temp,
    y_temp,
    snr_temp,
    test_size=0.17647,
    random_state=42,
    stratify=strat_temp
)


print("\nDataset split:")
print("Train:", X_train.shape, y_train.shape, snr_train.shape)
print("Validation:", X_val.shape, y_val.shape, snr_val.shape)
print("Test:", X_test.shape, y_test.shape, snr_test.shape)


print("\nExample training sample:")
print("Signal shape:", X_train[0].shape)
print("Label:", y_train[0])
print("SNR:", snr_train[0])


# =========================
# 6. 调制类型 → 数字标签
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


MOD_TO_IDX = {
    mod: idx
    for idx, mod in enumerate(MODULATIONS)
}


# 例如：
#
# GFSK → 5
# QPSK → 9


y_train_idx = np.array([
    MOD_TO_IDX[mod]
    for mod in y_train
])

y_val_idx = np.array([
    MOD_TO_IDX[mod]
    for mod in y_val
])

y_test_idx = np.array([
    MOD_TO_IDX[mod]
    for mod in y_test
])


# =========================
# 7. NumPy → PyTorch Tensor
# =========================

X_train_tensor = torch.from_numpy(
    X_train.astype(np.float32)
)

X_val_tensor = torch.from_numpy(
    X_val.astype(np.float32)
)

X_test_tensor = torch.from_numpy(
    X_test.astype(np.float32)
)


# 标签使用 long / int64
# 因为 CrossEntropyLoss 需要整数类别编号

y_train_tensor = torch.from_numpy(
    y_train_idx
).long()

y_val_tensor = torch.from_numpy(
    y_val_idx
).long()

y_test_tensor = torch.from_numpy(
    y_test_idx
).long()


# =========================
# 8. 创建 TensorDataset
# =========================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)


# =========================
# 9. 创建 DataLoader
# =========================

batch_size = 256


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# =========================
# 10. 检查第一个 batch
# =========================

x_batch, y_batch = next(
    iter(train_loader)
)


print("\nFirst batch:")
print("x batch shape:", x_batch.shape)
print("y batch shape:", y_batch.shape)
print("First label index:", y_batch[0].item())

