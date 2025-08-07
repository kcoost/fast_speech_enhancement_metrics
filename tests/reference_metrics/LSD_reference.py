# From https://github.com/urgent-challenge/urgent2025_challenge/blob/main/evaluation_metrics/calculate_intrusive_se_metrics.py#L66C1-L90C15
import librosa
import numpy as np
import torch
from fast_se_metrics.base import BaseMetric


def lsd_metric(ref, inf, fs, nfft=0.032, hop=0.016, p=2, eps=1.0e-08):
    """Calculate Log-Spectral Distance (LSD).

    Args:
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
        nfft (float): FFT length in seconds
        hop (float): hop length in seconds
        p (float): the order of norm
        eps (float): epsilon value for numerical stability
    Returns:
        mcd (float): LSD value between [0, +inf)
    """
    scaling_factor = np.sum(ref * inf) / (np.sum(inf**2) + eps)
    inf = inf * scaling_factor

    nfft = int(fs * nfft)
    hop = int(fs * hop)
    # T x F
    ref_spec = np.abs(librosa.stft(ref, hop_length=hop, n_fft=nfft)).T
    inf_spec = np.abs(librosa.stft(inf, hop_length=hop, n_fft=nfft)).T
    lsd = np.log(ref_spec**2 / ((inf_spec + eps) ** 2) + eps) ** p
    lsd = np.mean(np.mean(lsd, axis=1) ** (1 / p), axis=0)
    return lsd


class LSD_reference(BaseMetric):
    higher_is_better = False
    EXPECTED_SAMPLING_RATE = 16000

    def compute_metric(self, clean_speech: torch.Tensor, noisy_speech: torch.Tensor) -> list[dict[str, float]]:
        lsd_values = []
        for s, ns in zip(clean_speech, noisy_speech, strict=False):
            lsd = lsd_metric(s.numpy(), ns.numpy(), self.sample_rate)
            lsd_values.append({"LSD": float(lsd)})
        return lsd_values
