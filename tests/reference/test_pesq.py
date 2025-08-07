import pytest
import torch
from gpu_speech_metrics import PESQ
from gpu_speech_metrics.base import BaseMetric
from pesq import pesq as pesq_metric_reference
from torch_pesq import PesqLoss


class PESQ_reference(BaseMetric):
    higher_is_better = True
    EXPECTED_SAMPLING_RATE = 16000

    def compute_metric(self, clean_speech: torch.Tensor, noisy_speech: torch.Tensor) -> list[dict[str, float]]:
        pesqs = []
        for s, ns in zip(clean_speech, noisy_speech, strict=False):
            pesq = pesq_metric_reference(16000, s.numpy(), ns.numpy(), mode="wb")
            pesqs.append({"PESQ": pesq})
        return pesqs


class PESQ_torch(BaseMetric):
    higher_is_better = True
    EXPECTED_SAMPLING_RATE = 16000

    def __init__(self, sample_rate: int = 16000, device: str = "cpu"):
        super().__init__(sample_rate, device == "cuda")
        self.pesq_loss = PesqLoss(1.0, sample_rate=16000)
        self.pesq_loss.to(device)

    def compute_metric(self, clean_speech: torch.Tensor, noisy_speech: torch.Tensor) -> list[dict[str, float]]:
        if self.device == "cuda":
            clean_speech = clean_speech.to("cuda")
            noisy_speech = noisy_speech.to("cuda")

        with torch.inference_mode():
            pesqs = self.pesq_loss.mos(clean_speech, noisy_speech)
        return [{"PESQ": pesq.item()} for pesq in pesqs]


def test_pesq(speech_data):
    speeches = speech_data["speech"]
    noisy_speeches = speech_data["noisy_speech"]

    reference_pesqs = []
    for speech, noisy_speech in zip(speeches, noisy_speeches, strict=False):
        reference_pesq = pesq_metric_reference(16000, speech.numpy(), noisy_speech.numpy(), mode="wb")
        reference_pesqs.append(reference_pesq)

    pesq_metric = PESQ()
    pesq_loss = PesqLoss(1.0, sample_rate=16000)

    pesqs = pesq_metric(speeches, noisy_speeches)
    reference_torch_pesqs = pesq_loss.mos(speeches, noisy_speeches)

    for pesq, reference_torch_pesq, reference_pesq in zip(pesqs, reference_torch_pesqs, reference_pesqs, strict=False):
        assert pesq["PESQ"] == pytest.approx(reference_pesq, abs=0.1)
        assert pesq["PESQ"] == pytest.approx(reference_torch_pesq, abs=1e-7)
