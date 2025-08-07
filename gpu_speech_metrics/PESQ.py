import torch
import numpy as np

from scipy.signal import butter
from torchaudio.functional import lfilter
from torchaudio.transforms import Spectrogram

from gpu_speech_metrics.utils.bark import BarkFilterBank
from gpu_speech_metrics.utils.loudness import Loudness
from gpu_speech_metrics.base import BaseMetric


class PESQ(BaseMetric):
    higher_is_better = True
    EXPECTED_SAMPLING_RATE = 16000

    """Perceptual Evaluation of Speech Quality

    Implementation of the PESQ score in the PyTorch framework, closely following the ITU P.862
    reference. There are two mayor difference:

      1. no time alignment
      2. energy normalization uses an IIR filter

    Parameters
    ----------
    factor : float
        Scaling of the loss function
    sample_rate : int
        Sampling rate of the time signal, re-samples if different from 16kHz
    nbarks : int
        Number of bark bands
    win_length : int
        Window size used in the STFT
    n_fft : int
        Number of frequency bins
    hop_length : int
        Distance between different frames

    Attributes
    ----------
    to_spec : torch.nn.Module
        Perform a Short-Time Fourier Transformation on the time signal returning the power spectral
        density
    fbank : torch.nn.Module
        Apply a Bark scaling to the power distribution
    loudness : torch.nn.Module
        Estimate perceived loudness of the Bark scaled spectrogram
    power_filter : TensorType
        IIR filter coefficients to calculate power in 325Hz to 3.25kHz band
    pre_filter : TensorType
        Pre-empasize filter, applied to reference and degraded signal
    """

    def __init__(self, sample_rate: int = 16000, use_gpu: bool = False):
        super().__init__(sample_rate, use_gpu)
        nbarks = 49
        win_length = 512
        n_fft = 512
        hop_length = 256

        # PESQ specifications state 32ms, 50% overlap, Hamming window
        self.to_spec = Spectrogram(
            win_length=win_length,
            n_fft=n_fft,
            hop_length=hop_length,
            window_fn=torch.hann_window,
            power=2,
            normalized=False,
            center=False,
        ).to(self.device)

        # use a Bark filterbank to model perceived frequency resolution
        self.filter_bank = BarkFilterBank(n_fft // 2, nbarks, device=self.device)

        # set up loudness degation and calibration
        self.loudness = Loudness(nbarks, device=self.device)

        # design IIR bandpass filter for power degation between 325Hz to 3.25kHz
        out = np.asarray(butter(5, [325, 3250], fs=16000, btype="band"))
        self.power_filter = torch.as_tensor(out, device=self.device, dtype=torch.float32)

        # use IIR filter for pre-emphasize
        self.pre_filter = torch.tensor(
            [[2.740826, -5.4816519, 2.740826], [1.0, -1.9444777, 0.94597794]],
            device=self.device,
            dtype=torch.float32,
        )

        self.taper_weights = torch.linspace(0, 15, 16, device=self.device)[1:] / 16.0

    def align_level(self, speech: torch.Tensor) -> torch.Tensor:
        # Align power to 10**7 for band 325 to 3.25kHz
        filtered_speech = lfilter(speech, self.power_filter[1], self.power_filter[0], clamp=False)

        # calculate power with weird bugs in reference implementation
        power = (filtered_speech.square()).sum(dim=1, keepdim=True) / (filtered_speech.shape[1] + 5120) / 1.04684

        # align power
        speech = speech * (10**7 / power).sqrt()

        return speech

    def pre_emphasize(self, speech: torch.Tensor) -> torch.Tensor:
        # This pre-emphasize filter is also applied in the reference implementation.
        # The filter coefficients are taken from the reference.

        speech[:, :15] *= self.taper_weights
        speech[:, -15:] *= torch.flip(self.taper_weights, dims=(0,))

        speech = lfilter(speech, self.pre_filter[1], self.pre_filter[0], clamp=False)

        return speech

    @staticmethod
    def equalize_ranges(clean_speech: torch.Tensor, noisy_speech: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_value = torch.max(
            torch.amax(clean_speech.abs(), dim=1, keepdim=True),
            torch.amax(noisy_speech.abs(), dim=1, keepdim=True),
        )
        return clean_speech / max_value, noisy_speech / max_value

    def get_bark_bands(self, speech: torch.Tensor) -> torch.Tensor:
        speech = self.align_level(speech)
        speech = self.pre_emphasize(speech)

        # do weird alignments with reference implementation
        pad_amount = speech.shape[1] % 256
        if pad_amount > 0:
            speech = torch.nn.functional.pad(speech, (0, pad_amount))

        # calculate spectrogram for ref and degated speech
        speech = self.to_spec(speech).swapaxes(1, 2)

        # we won't use energy feature
        speech[:, :, 0] = 0.0

        # calculate power spectrum in bark scale and hearing threshold
        speech = self.filter_bank(speech)
        return speech

    def equalize_bark_bands(
        self, clean_bark_bands: torch.Tensor, noisy_bark_bands: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # degate silent frames
        frame_is_silent = self.loudness.audible_frame_power(clean_bark_bands, 1e2) < 1e7

        mean_clean_band_power = self.loudness.mean_audible_band_power(clean_bark_bands, frame_is_silent)
        mean_noisy_band_power = self.loudness.mean_audible_band_power(noisy_bark_bands, frame_is_silent)

        band_power_ratio = (mean_noisy_band_power + 1000) / (mean_clean_band_power + 1000)
        band_power_ratio = band_power_ratio.clamp(min=0.01, max=100.0)

        equalized_clean_bark_bands = band_power_ratio.unsqueeze(1) * clean_bark_bands

        # normalize power of degated signal, averaged over bands
        frame_power_ratio = (self.loudness.audible_frame_power(equalized_clean_bark_bands, 1) + 5e3) / (
            self.loudness.audible_frame_power(noisy_bark_bands, 1) + 5e3
        )

        frame_power_ratio[:, 1:] = 0.8 * frame_power_ratio[:, 1:] + 0.2 * frame_power_ratio[:, :-1]
        frame_power_ratio = frame_power_ratio.clamp(min=3e-4, max=5.0)

        equalized_noisy_bark_bands = frame_power_ratio * noisy_bark_bands

        return equalized_clean_bark_bands, equalized_noisy_bark_bands

    def get_overlapping_sums(self, disturbance: torch.Tensor) -> torch.Tensor:
        frames = disturbance.unfold(1, size=20, step=10)
        psqm = frames.pow(6).mean(dim=2).pow(1 / 6)
        distance = psqm.square().mean(dim=1).sqrt()
        return distance

    def get_disturbances(
        self, clean_speech: torch.Tensor, noisy_speech: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = clean_speech.shape[0]
        clean_speech = torch.atleast_2d(clean_speech)
        noisy_speech = torch.atleast_2d(noisy_speech)

        clean_speech, noisy_speech = self.equalize_ranges(clean_speech, noisy_speech)

        speech = torch.cat([clean_speech, noisy_speech], dim=0)
        bark_bands = self.get_bark_bands(speech)
        clean_bark_bands = bark_bands[:batch_size]
        noisy_bark_bands = bark_bands[batch_size:]

        equalized_clean_bark_bands, equalized_noisy_bark_bands = self.equalize_bark_bands(
            clean_bark_bands, noisy_bark_bands
        )

        equalized_bark_bands = torch.cat([equalized_clean_bark_bands, equalized_noisy_bark_bands], dim=0)
        loudness = self.loudness.loudness(equalized_bark_bands)
        clean_loudness = loudness[:batch_size]
        noisy_loudness = loudness[batch_size:]

        # calculate disturbance
        deadzone = 0.25 * torch.min(clean_loudness, noisy_loudness).squeeze(-1)
        disturbance = noisy_loudness - clean_loudness
        disturbance = disturbance.sign() * (disturbance.abs() - deadzone).clamp(min=0)

        """
        if deg > 1.25*ref: deg - 1.25*ref
        if ref < deg < 1.25*ref: 0
        if ref > 1.25*deg: 1.25*deg - ref
        if deg < ref < 1.25*deg: 0
        """

        # symmetrical disturbance
        symmetric_disturbance = self.filter_bank.weighted_norm(disturbance, p=2)
        symmetric_disturbance = symmetric_disturbance.clamp(min=1e-20)

        # asymmetrical disturbance
        asymmetric_scaling = ((equalized_noisy_bark_bands + 50.0) / (equalized_clean_bark_bands + 50.0)).pow(1.2)
        asymmetric_scaling[asymmetric_scaling < 3.0] = 0.0
        asymmetric_scaling = asymmetric_scaling.clamp(max=12.0)

        asymmetric_disturbance = self.filter_bank.weighted_norm(disturbance * asymmetric_scaling, p=1)
        asymmetric_disturbance = asymmetric_disturbance.clamp(min=1e-20)

        # weighting
        weight = ((self.loudness.audible_frame_power(equalized_clean_bark_bands, 1) + 1e5) / 1e7).pow(0.04)
        symmetric_disturbance = (symmetric_disturbance / weight.squeeze(-1)).clamp(max=45.0)
        asymmetric_disturbance = (asymmetric_disturbance / weight.squeeze(-1)).clamp(max=45.0)

        # calculate overlapping sums
        symmetric_distance = self.get_overlapping_sums(symmetric_disturbance)
        asymmetric_distance = self.get_overlapping_sums(asymmetric_disturbance)

        return symmetric_distance, asymmetric_distance

    def compute_metric(
        self, clean_speech: torch.Tensor | None, denoised_speech: torch.Tensor
    ) -> list[dict[str, float]]:
        assert clean_speech is not None
        with torch.inference_mode():
            symmetric_distance, asymmetric_distance = self.get_disturbances(clean_speech, denoised_speech)

            # calculate MOS as combination of symmetric and asymmetric distance
            mos = 4.5 - 0.1 * symmetric_distance - 0.0309 * asymmetric_distance

            # apply compression curve to have MOS in proper range
            mos = 0.999 + 4 / (1 + torch.exp(-1.3669 * mos + 3.8224))

            return [{"PESQ": m.item()} for m in mos]
