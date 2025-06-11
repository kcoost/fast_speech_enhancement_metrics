import pytest
from benchmarking.dataloading import load_data

SAMPLE_DURATION = 16
NUM_SAMPLES = 8

@pytest.fixture
def speech_data():
    speech, noisy_speech, snr = load_data(SAMPLE_DURATION, NUM_SAMPLES)
    return {"speech": speech, "noisy_speech": noisy_speech, "snr": snr}

@pytest.fixture
def high_snr_speech_data():
    speech, noisy_speech, snr = load_data(SAMPLE_DURATION, NUM_SAMPLES, SNR_high=10, SNR_low=10)
    return {"speech": speech, "noisy_speech": noisy_speech, "snr": snr}

@pytest.fixture
def low_snr_speech_data():
    speech, noisy_speech, snr = load_data(SAMPLE_DURATION, NUM_SAMPLES, SNR_high=-5, SNR_low=-5)
    return {"speech": speech, "noisy_speech": noisy_speech, "snr": snr}