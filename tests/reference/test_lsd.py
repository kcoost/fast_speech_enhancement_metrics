import pytest
import librosa
import numpy as np
from fast_se_metrics import LSD


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


def test_lsd(speech_data):
    clean_speeches = speech_data["speech"]
    noisy_speeches = speech_data["noisy_speech"]

    lsd = LSD(16000)

    reference_results = []
    for clean_speech, noisy_speech in zip(clean_speeches, noisy_speeches, strict=False):
        reference_result = lsd_metric(clean_speech.numpy(), noisy_speech.numpy(), 16000)
        reference_results.append(reference_result)

    results = lsd(clean_speeches, noisy_speeches)

    for reference_result, result in zip(reference_results, results, strict=False):
        assert reference_result == pytest.approx(result["LSD"], 1e-5)
