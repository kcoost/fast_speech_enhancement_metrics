import torch
from pystoi import stoi as stoi_reference
from gpu_speech_metrics.base import BaseMetric


class STOI_reference(BaseMetric):
    higher_is_better = True
    EXPECTED_SAMPLING_RATE = 10000

    def compute_metric(self, clean_speech: torch.Tensor, noisy_speech: torch.Tensor) -> list[dict[str, float]]:
        stois = []
        for speech, noisy_speech in zip(clean_speech, noisy_speech):
            stoi = stoi_reference(speech.numpy(), noisy_speech.numpy(), 10000, extended=False)
            estoi = stoi_reference(speech.numpy(), noisy_speech.numpy(), 10000, extended=True)
            stois.append({"STOI": stoi, "ESTOI": estoi})
        return stois