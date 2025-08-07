import torch
from torchaudio.transforms import Resample
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    higher_is_better: bool
    EXPECTED_SAMPLING_RATE: int

    def __init__(self, sample_rate: int = 16000, use_gpu: bool = False):
        self.sample_rate = sample_rate
        self.device = "cuda" if use_gpu else "cpu"
        self.resampler = Resample(sample_rate, self.EXPECTED_SAMPLING_RATE)
        self.resampler.to(self.device)

    def prepare_audio(self, audio: torch.Tensor) -> torch.Tensor:
        audio = torch.atleast_2d(audio)
        audio = audio.to(self.device)
        if self.sample_rate != self.EXPECTED_SAMPLING_RATE:
            audio = self.resampler(audio)
        return audio

    def prepare_inputs(
        self, clean_speech: torch.Tensor | None, denoised_speech: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if clean_speech is not None and clean_speech.shape != denoised_speech.shape:
            raise Exception("`clean_speech` and `denoised_speech` should have the same shape.")

        if clean_speech is not None:
            clean_speech = self.prepare_audio(clean_speech)
        denoised_speech = self.prepare_audio(denoised_speech)

        return clean_speech, denoised_speech

    @abstractmethod
    def compute_metric(
        self, clean_speech: torch.Tensor | None, denoised_speech: torch.Tensor
    ) -> list[dict[str, float]]:
        raise NotImplementedError

    def __call__(self, clean_speech: torch.Tensor | None, denoised_speech: torch.Tensor) -> list[dict[str, float]]:
        clean_speech, denoised_speech = self.prepare_inputs(clean_speech, denoised_speech)
        return self.compute_metric(clean_speech, denoised_speech)
