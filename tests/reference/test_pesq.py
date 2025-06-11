import pytest
from gpu_speech_metrics import PESQ
from pesq import pesq as pesq_metric_reference
from torch_pesq import PesqLoss

def test_pesq(speech_data):
    speeches = speech_data["speech"]
    noisy_speeches = speech_data["noisy_speech"]

    reference_pesqs = []
    for speech, noisy_speech in zip(speeches, noisy_speeches):
        reference_pesq = pesq_metric_reference(16000, speech.numpy(), noisy_speech.numpy(), mode="wb")
        reference_pesqs.append(reference_pesq)
    
    pesq_metric = PESQ()
    pesq_loss = PesqLoss(1.0, sample_rate=16000)

    pesqs = pesq_metric(speeches, noisy_speeches)
    reference_torch_pesqs = pesq_loss.mos(speeches, noisy_speeches)

    for pesq, reference_torch_pesq, reference_pesq in zip(pesqs, reference_torch_pesqs, reference_pesqs):
        assert pesq["PESQ"] == pytest.approx(reference_pesq, abs=0.1)
        assert pesq["PESQ"] == pytest.approx(reference_torch_pesq, abs=1e-7)
