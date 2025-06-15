import torch
from gpu_speech_metrics.base import BaseMetric

import torchmetrics

class SDR_reference(BaseMetric):
    higher_is_better = True
    EXPECTED_SAMPLING_RATE = 16000
    # def __init__(self, sample_rate: int, device: str = "cpu"):
    #     super().__init__(sample_rate=sample_rate, device=device)
    #     self.sdr = SignalDistortionRatio()
    #     self.sdr.to(device)

    def compute_metric(self, clean_speech: torch.Tensor, noisy_speech: torch.Tensor) -> list[dict[str, float]]:
        sdrs = torchmetrics.functional.audio.signal_distortion_ratio(noisy_speech, clean_speech)
        return [{"SDR": sdr.item()} for sdr in sdrs]