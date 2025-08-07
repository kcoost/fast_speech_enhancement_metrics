import pytest
from gpu_speech_metrics import LSD, STOI, PESQ, SDR, DNSMOS, SpeechBERTScore


METRICS = [LSD, STOI, PESQ, SDR, DNSMOS, SpeechBERTScore]


@pytest.mark.parametrize("metric_class", METRICS)
def test_device(speech_data, metric_class):
    speech = speech_data["speech"]
    noisy_speech = speech_data["noisy_speech"]

    metric = metric_class()
    cpu_results = metric(speech, noisy_speech)

    metric = metric_class(device="cuda")
    gpu_results = metric(speech, noisy_speech)

    for cpu_result, gpu_result in zip(cpu_results, gpu_results, strict=False):
        if metric_class == SDR:
            assert cpu_result == pytest.approx(gpu_result, abs=1e-1)
        else:
            assert cpu_result == pytest.approx(gpu_result, abs=5e-3)
