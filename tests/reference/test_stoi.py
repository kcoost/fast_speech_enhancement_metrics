import pytest
from fast_se_metrics import STOI
from pystoi import stoi as stoi_reference


def test_stoi(speech_data):
    speeches = speech_data["speech"]
    noisy_speeches = speech_data["noisy_speech"]

    stoi_metric = STOI(sample_rate=10000)
    stoi_results = stoi_metric(speeches, noisy_speeches)

    stoi_results_reference = []
    estoi_results_reference = []
    for speech, noisy_speech in zip(speeches, noisy_speeches, strict=False):
        stoi_result = stoi_reference(speech.numpy(), noisy_speech.numpy(), 10000, extended=False)
        estoi_result = stoi_reference(speech.numpy(), noisy_speech.numpy(), 10000, extended=True)
        stoi_results_reference.append(stoi_result)
        estoi_results_reference.append(estoi_result)

    for stoi_result, stoi_result_reference, estoi_result_reference in zip(
        stoi_results, stoi_results_reference, estoi_results_reference, strict=False
    ):
        assert stoi_result["STOI"] == pytest.approx(stoi_result_reference, abs=5e-4)
        assert stoi_result["ESTOI"] == pytest.approx(estoi_result_reference, abs=5e-4)
