import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

MODULATIONS = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
MOD_TO_IDX = {m: i for i, m in enumerate(MODULATIONS)}
SNR_VALUES = list(range(-20, 20, 2))  # -20 to +18 dB in steps of 2


def compute_expert_features(x):
    """
    Compute instantaneous amplitude and phase from IQ samples.
    Input:  x shape (..., 2, 128) — I and Q channels
    Output: shape (..., 4, 128) — [I, Q, amplitude, phase]
    """
    i = x[..., 0:1, :]   # (..., 1, 128)
    q = x[..., 1:2, :]   # (..., 1, 128)
    amplitude = np.sqrt(i ** 2 + q ** 2)
    phase = np.arctan2(q, i) / np.pi  # normalize to [-1, 1]
    return np.concatenate([x, amplitude, phase], axis=-2)  # (..., 4, 128)


class RadioMLDataset(Dataset):
    def __init__(self, path, snr_range=None, augment=False, expert_features=False):
        """
        Args:
            path:            path to RML2016.10a_dict.pkl
            snr_range:       optional (min_snr, max_snr) to filter by SNR
            augment:         if True, apply random phase rotation during __getitem__
            expert_features: if True, append instantaneous amplitude & phase → (4, 128)
        """
        with open(path, 'rb') as f:
            raw = pickle.load(f, encoding='latin1')

        self.augment = augment
        self.expert_features = expert_features

        samples, labels, snrs = [], [], []
        for (mod, snr), arr in raw.items():
            if snr_range is not None and not (snr_range[0] <= snr <= snr_range[1]):
                continue
            samples.append(arr)
            labels.append(np.full(len(arr), MOD_TO_IDX[mod]))
            snrs.append(np.full(len(arr), snr))

        data = np.concatenate(samples).astype(np.float32)  # (N, 2, 128)

        if expert_features:
            data = compute_expert_features(data).astype(np.float32)  # (N, 4, 128)

        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(np.concatenate(labels)).long()
        self.snrs = np.concatenate(snrs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].clone()
        if self.augment:
            # random phase rotation: multiply IQ by e^(j*theta)
            theta = torch.empty(1).uniform_(0, 2 * torch.pi)
            cos_t, sin_t = theta.cos(), theta.sin()
            i = x[0].clone()
            q = x[1].clone()
            x[0] = cos_t * i - sin_t * q
            x[1] = sin_t * i + cos_t * q
        return x, self.labels[idx]


def get_dataloaders(path, val_split=0.1, test_split=0.1, batch_size=256,
                    seed=42, augment=False, expert_features=False):
    dataset = RadioMLDataset(path, augment=augment, expert_features=expert_features)
    n = len(dataset)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    # val/test never augmented
    train_ds.dataset.augment = augment

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
