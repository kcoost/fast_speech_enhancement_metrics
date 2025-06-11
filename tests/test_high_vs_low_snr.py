import pytest
from gpu_speech_metrics import LSD, STOI, PESQ, SDR, DNSMOS, SpeechBERTScore


METRICS = [LSD, STOI, PESQ, SDR, DNSMOS, SpeechBERTScore]

@pytest.mark.parametrize("metric_class", METRICS)
def test_differentiation(high_snr_speech_data, low_snr_speech_data, metric_class):
    high_snr_speech = high_snr_speech_data["speech"]
    low_snr_speech = low_snr_speech_data["speech"]

    high_snr_noisy_speech = high_snr_speech_data["noisy_speech"]
    low_snr_noisy_speech = low_snr_speech_data["noisy_speech"]

    metric = metric_class(sample_rate=16000)
    high_snr_results = metric(high_snr_speech, high_snr_noisy_speech)
    low_snr_results = metric(low_snr_speech, low_snr_noisy_speech)

    for high_snr_result, low_snr_result in zip(high_snr_results, low_snr_results):
        if metric.higher_is_better:
            for key in high_snr_result:
                assert high_snr_result[key] > low_snr_result[key]
        else:
            for key in high_snr_result:
                assert high_snr_result[key] < low_snr_result[key]
