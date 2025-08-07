import pytest
from tests.reference_metrics.SpeechBERTScore_reference import SpeechBERTScore_reference
from fast_se_metrics import SpeechBERTScore


def speechbertscore_metric(ref, inf, sample_rate=16000, device="cuda"):
    """Calculate SpeechBERTScore using reference implementation.

    Args:
        ref (torch.Tensor): reference signal (time,)
        inf (torch.Tensor): enhanced signal (time,)
        sample_rate (int): sampling rate
        device (str): device to use
    Returns:
        speechbertscore (float): SpeechBERTScore F1 value
    """
    speechbertscore = SpeechBERTScore_reference(sample_rate=sample_rate, device=device)
    result = speechbertscore.compute_metric(ref.unsqueeze(0), inf.unsqueeze(0))
    return result[0]["SpeechBERTScore"]


def test_speechbertscore(speech_data):
    clean_speeches = speech_data["speech"]
    noisy_speeches = speech_data["noisy_speech"]

    speechbertscore = SpeechBERTScore(16000, device="cuda")

    reference_results = []
    for clean_speech, noisy_speech in zip(clean_speeches, noisy_speeches, strict=False):
        reference_result = speechbertscore_metric(clean_speech, noisy_speech)
        reference_results.append(reference_result)

    results = speechbertscore(clean_speeches, noisy_speeches)

    for reference_result, result in zip(reference_results, results, strict=False):
        assert reference_result == pytest.approx(result["SpeechBERTScore"], abs=1e-5)
