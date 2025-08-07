# Based on https://github.com/urgent-challenge/urgent2025_challenge/blob/main/evaluation_metrics/calculate_intrusive_se_metrics.py
import torch
from gpu_speech_metrics.base import BaseMetric


class LSD(BaseMetric):
    higher_is_better = False
    EXPECTED_SAMPLING_RATE = 16000

    def __init__(self, sample_rate: int = 16000, use_gpu: bool = False):
        super().__init__(sample_rate, use_gpu)
        self.nfft = int(self.EXPECTED_SAMPLING_RATE * 0.032)
        self.hop = int(self.EXPECTED_SAMPLING_RATE * 0.016)
        self.p = 2
        self.eps = 1e-8
        self.window = torch.hann_window(self.nfft, device=self.device)

    def torch_stft(self, signal: torch.Tensor) -> torch.Tensor:
        # Gives the same result as np.abs(librosa.stft(ref, hop_length=hop, n_fft=nfft))

        spectrogram = torch.stft(
            signal,
            n_fft=self.nfft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        return spectrogram.abs()

    def compute_metric(
        self, clean_speech: torch.Tensor | None, denoised_speech: torch.Tensor
    ) -> list[dict[str, float]]:
        assert clean_speech is not None
        batch_size = clean_speech.shape[0]
        scaling_factor = torch.sum(clean_speech * denoised_speech, dim=1, keepdim=True) / (
            torch.sum(denoised_speech**2, dim=1, keepdim=True) + self.eps
        )
        denoised_speech = denoised_speech * scaling_factor

        speech = torch.cat([clean_speech, denoised_speech], dim=0)
        speech_spectrogram = self.torch_stft(speech)
        clean_speech_spectrogram = speech_spectrogram[:batch_size]
        denoised_speech_spectrogram = speech_spectrogram[batch_size:]

        lsds = torch.log(
            clean_speech_spectrogram.square() / ((denoised_speech_spectrogram + self.eps).square()) + self.eps
        ).pow(self.p)
        lsds = lsds.mean(dim=1).pow(1 / self.p).mean(dim=1)

        return [{"LSD": lsd.item()} for lsd in lsds]
