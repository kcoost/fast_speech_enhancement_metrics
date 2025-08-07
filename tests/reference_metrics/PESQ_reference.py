# From https://github.com/ludlows/PESQ
import torch
from fast_se_metrics.base import BaseMetric

from pesq import pesq_batch as pesq_metric_reference
from torch_pesq import PesqLoss


class PESQ_reference_pesq(BaseMetric):
    higher_is_better = True
    EXPECTED_SAMPLING_RATE = 16000

    def compute_metric(self, clean_speech: torch.Tensor, noisy_speech: torch.Tensor) -> list[dict[str, float]]:
        pesqs = pesq_metric_reference(16000, clean_speech.numpy(), noisy_speech.numpy(), mode="wb")
        return [{"PESQ": pesq} for pesq in pesqs]


class PESQ_reference_torch_pesq(BaseMetric):
    higher_is_better = True
    EXPECTED_SAMPLING_RATE = 16000

    def __init__(self, sample_rate: int = 16000, use_gpu: bool = False):
        super().__init__(sample_rate, use_gpu)
        self.pesq_loss = PesqLoss(1.0, sample_rate=16000)
        self.pesq_loss.to(self.device)

    def compute_metric(self, clean_speech: torch.Tensor, noisy_speech: torch.Tensor) -> list[dict[str, float]]:
        if self.device == "cuda":
            clean_speech = clean_speech.to("cuda")
            noisy_speech = noisy_speech.to("cuda")

        with torch.inference_mode():
            pesqs = self.pesq_loss.mos(clean_speech, noisy_speech)
        return [{"PESQ": pesq.item()} for pesq in pesqs]
