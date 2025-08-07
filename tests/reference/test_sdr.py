import pytest
import torchmetrics.functional.audio
from gpu_speech_metrics import SDR


def sdr_metric(ref, inf):
    """Calculate Signal-to-Distortion Ratio (SDR) using torchmetrics.

    Args:
        ref (torch.Tensor): reference signal (time,)
        inf (torch.Tensor): enhanced signal (time,)
    Returns:
        sdr (float): SDR value in dB
    """
    # torchmetrics expects (batch, time) and preds, target order
    ref = ref.unsqueeze(0)  # Add batch dimension
    inf = inf.unsqueeze(0)  # Add batch dimension
    sdr = torchmetrics.functional.audio.signal_distortion_ratio(inf, ref)
    return sdr.item()


def test_sdr(speech_data):
    clean_speeches = speech_data["speech"]
    noisy_speeches = speech_data["noisy_speech"]

    sdr = SDR(16000)

    reference_results = []
    for clean_speech, noisy_speech in zip(clean_speeches, noisy_speeches, strict=False):
        reference_result = sdr_metric(clean_speech, noisy_speech)
        reference_results.append(reference_result)

    results = sdr(clean_speeches, noisy_speeches)

    for reference_result, result in zip(reference_results, results, strict=False):
        assert reference_result == pytest.approx(result["SDR"], abs=1e-2)
