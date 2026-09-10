import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .dataset import MODULATIONS, SNR_VALUES, RadioMLDataset
from torch.utils.data import DataLoader


def accuracy_vs_snr(model, data_path, device='cuda', batch_size=256, expert_features=False):
    """Compute per-SNR accuracy — the key evaluation plot."""
    model.eval()
    results = {}

    for snr in SNR_VALUES:
        ds = RadioMLDataset(data_path, snr_range=(snr, snr), expert_features=expert_features)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        correct = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
        results[snr] = correct / len(ds)

    return results


def plot_accuracy_vs_snr(results, save_path=None):
    snrs = sorted(results.keys())
    accs = [results[s] * 100 for s in snrs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(snrs, accs, marker='o', linewidth=2, color='steelblue', markersize=5)
    ax.axhline(100 / 11, color='gray', linestyle='--', linewidth=1, label='Random chance (9.1%)')
    ax.set_xlabel('SNR (dB)', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Modulation Classification Accuracy vs. SNR', fontsize=14)
    ax.set_xticks(snrs)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d%%'))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_confusion_matrix(model, data_path, snr_min=0, device='cuda',
                          batch_size=256, save_path=None):
    """Confusion matrix at SNR >= snr_min (high-SNR regime)."""
    ds = RadioMLDataset(data_path, snr_range=(snr_min, 30))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            all_preds.append(model(x).argmax(1).cpu().numpy())
            all_labels.append(y.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    cm = confusion_matrix(labels, preds, normalize='true')
    fig, ax = plt.subplots(figsize=(11, 9))
    disp = ConfusionMatrixDisplay(cm, display_labels=MODULATIONS)
    disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='.2f')
    ax.set_title(f'Confusion Matrix (SNR ≥ {snr_min} dB)', fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
