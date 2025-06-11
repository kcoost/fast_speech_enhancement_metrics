# Based on https://github.com/urgent-challenge/urgent2025_challenge/blob/main/evaluation_metrics/calculate_intrusive_se_metrics.py
import torch
from gpu_speech_metrics.base import BaseMetric

class LSD(BaseMetric):
    higher_is_better = False
    EXPECTED_SAMPLING_RATE = 16000

    def __init__(self, sample_rate: int = 16000, device: str = "cpu"):
        super().__init__(sample_rate, device)
        self.nfft = int(self.EXPECTED_SAMPLING_RATE * 0.032)
        self.hop = int(self.EXPECTED_SAMPLING_RATE * 0.016)
        self.p = 2
        self.eps = 1.0e-08

    def torch_stft(self, signal: torch.Tensor) -> torch.Tensor:
        # Gives the same result as np.abs(librosa.stft(ref, hop_length=hop, n_fft=nfft))
        window = torch.hann_window(self.nfft, device=self.device)
        spectrogram = torch.stft(signal,
                        n_fft=self.nfft,
                        hop_length=self.hop,
                        window=window,
                        center=True,
                        pad_mode="constant",
                        return_complex=True)
        return spectrogram.abs()
    
    def compute_metric(self, clean_speech: torch.Tensor, denoised_speech: torch.Tensor) -> list[dict[str, float]]:
        scaling_factor = torch.sum(clean_speech * denoised_speech, dim=1, keepdim=True) / (torch.sum(denoised_speech**2, dim=1, keepdim=True) + self.eps)
        denoised_speech = denoised_speech * scaling_factor

        clean_speech_spectrogram = self.torch_stft(clean_speech)
        denoised_speech_spectrogram = self.torch_stft(denoised_speech)

        lsds = torch.log(clean_speech_spectrogram**2 / ((denoised_speech_spectrogram + self.eps) ** 2) + self.eps) ** self.p
        lsds = lsds.mean(dim=1).pow(1/self.p).mean(dim=1)

        return [{"LSD": lsd.item()} for lsd in lsds]
