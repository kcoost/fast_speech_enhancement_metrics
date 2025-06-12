import numpy as np
import warnings

import numpy as np
import torch
from gpu_speech_metrics.base import BaseMetric

class STOI(BaseMetric):
    higher_is_better = True
    EXPECTED_SAMPLING_RATE = 10000

    def __init__(self, sample_rate: int = 10000, device: str = "cpu"):
        super().__init__(sample_rate, device)
        self.sampling_frequency = self.EXPECTED_SAMPLING_RATE
        self.win_length = 256
        self.hop_length = self.win_length // 2
        self.n_fft = 512
        self.num_octave_bands = 15
        self.min_frequency = 150 # Center frequency of 1st octave band (Hz)
        self.octave_band_matrix = self.get_octave_band_matrix()
        self.N = 30 # N. frames for intermediate intelligibility
        self.beta = -15. # Lower SDR bound
        self.dynamic_range = 40
    
    def get_octave_band_matrix(self) -> torch.Tensor:
        # Computes the 1/3 octave band matrix
        num_frequencies = self.n_fft//2 + 1
        
        frequencies = torch.linspace(0, self.sampling_frequency//2, num_frequencies, dtype=torch.float64)
        
        # Calculate center frequencies and frequency bounds
        band_idx = torch.arange(self.num_octave_bands, dtype=torch.float64) # important for precision
        frequencies_low = self.min_frequency * torch.pow(2.0, (2 * band_idx - 1) / 6)
        frequencies_high = self.min_frequency * torch.pow(2.0, (2 * band_idx + 1) / 6)
        
        octave_band_matrix = torch.zeros((self.num_octave_bands, num_frequencies), dtype=torch.float64)        
        for i in range(self.num_octave_bands):
            # Match 1/3 oct band freq with fft frequency bin
            idx_bin_low = torch.argmin((frequencies - frequencies_low[i]).abs())
            idx_bin_high = torch.argmin((frequencies - frequencies_high[i]).abs())

            frequencies_low[i] = frequencies[idx_bin_low]
            frequencies_high[i] = frequencies[idx_bin_high]
            
            octave_band_matrix[i, idx_bin_low:idx_bin_high] = 1
        return octave_band_matrix.to(torch.float32)

    def stft(self, speech: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.win_length+1, dtype=speech.dtype, device=speech.device)[1:]

        spectrogram = torch.stft(
            speech,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            normalized=False,
            return_complex=True,
            onesided=True
        )
        spectrogram = spectrogram.abs().square()
        # for i in range(len(lengths)):
        #     spectrogram_length = 1 + (lengths[i].item() - self.n_fft) // self.hop_length
        #     spectrogram[i, :, spectrogram_length:] = 0
        spectrogram_lengths = 1 + (lengths - self.n_fft) // self.hop_length
        time_idx = torch.arange(spectrogram.shape[-1], device=spectrogram.device)
        mask = time_idx.unsqueeze(0) >= spectrogram_lengths.unsqueeze(1)
        spectrogram.masked_fill_(mask.unsqueeze(1), 0)
        return spectrogram

    def overlap_and_add(self, frames: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        final_lengths = (lengths + 1) * self.hop_length
        max_length = int(torch.max(final_lengths).item())
        batch_size = len(final_lengths)
        
        signal = torch.zeros((batch_size, max_length), dtype=frames.dtype, device=frames.device)

        for i, frame in enumerate(frames.split(lengths.tolist())):
            # Vectorized version of
            # for i in range(num_frames):
            #     signal[i * hop_length:i * hop_length + frame_length] += x_frames[i]
            idx = torch.arange(self.win_length, device=frames.device).unsqueeze(0) + self.hop_length * torch.arange(int(lengths[i].item()), device=frames.device).unsqueeze(1)
            signal[i] += signal[i].scatter_add(0, idx.flatten(), frame.flatten())
        return signal, final_lengths

    def remove_silent_frames(self, clean_speech: torch.Tensor, denoised_speech: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute Mask
        window = torch.hann_window(self.win_length + 1, dtype=clean_speech.dtype, device=clean_speech.device)[1:]
        window = window.unsqueeze(0).unsqueeze(0)

        # Create frames
        clean_frames = clean_speech.unfold(1, self.win_length, self.hop_length)
        clean_frames = clean_frames * window
        denoised_frames = denoised_speech.unfold(1, self.win_length, self.hop_length)
        denoised_frames = denoised_frames * window

        # Compute energies in dB
        clean_energies = 20 * torch.log10(torch.norm(clean_frames, dim=2) + 1e-9)

        # Find boolean mask of energies lower than dynamic_range dB
        # with respect to maximum clean speech energy frame
        mask = (torch.amax(clean_energies, dim=1, keepdim=True) - self.dynamic_range - clean_energies) < 0

        # Remove silent frames by masking
        clean_frames = clean_frames[mask]
        denoised_frames = denoised_frames[mask]
        num_frames = mask.sum(1)

        clean_silent, lengths_silent = self.overlap_and_add(clean_frames, num_frames)
        denoised_silent, _ = self.overlap_and_add(denoised_frames, num_frames)
        return clean_silent, denoised_silent, lengths_silent
    
    @staticmethod
    def normalize(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        x -= x.mean(dim=dim, keepdim=True)
        x += 1e-12*torch.randn_like(x)
        norm = torch.norm(x, p=2, dim=dim, keepdim=True)
        x /= norm
        return x
    
    def compute_segments(self, speech: torch.Tensor, lengths: torch.Tensor):
        spectrogram = self.stft(speech, lengths)
        tob = torch.sqrt(torch.bmm(self.octave_band_matrix.to(speech.device).unsqueeze(0).repeat(len(speech), 1, 1), spectrogram))
        segments = [tob[:, :, m:m + self.N] for m in range(max(tob.shape[2] - self.N + 1, 0))]
        return segments
    
    def equalize_clip(self, clean_segments: torch.Tensor, denoised_segments: torch.Tensor) -> torch.Tensor:
        # Find normalization constants and normalize
        normalization_consts = torch.norm(clean_segments, dim=3, keepdim=True) /(torch.norm(denoised_segments, dim=3, keepdim=True) + 1e-9)
        denoised_segments_normalized = denoised_segments * normalization_consts

        # Clip as described in [1]
        clip_value = 10 ** (-self.beta / 20)
        denoised_segments_normalized = torch.minimum(denoised_segments_normalized, clean_segments * (1 + clip_value))
        return denoised_segments_normalized

    @torch.no_grad()
    def compute_stoi(self, clean_speech: torch.Tensor, denoised_speech: torch.Tensor, extended: bool = False):
        clean_speech, denoised_speech, lengths_silent = self.remove_silent_frames(clean_speech, denoised_speech)

        clean_segments = self.compute_segments(clean_speech, lengths_silent)
        denoised_segments = self.compute_segments(denoised_speech, lengths_silent)

        # Ensure at least 30 frames for intermediate intelligibility
        if len(clean_segments) == 0:
            warnings.warn('Not enough non-silent frames. Please check your sound files', RuntimeWarning)
            return 0
        
        clean_segments = torch.stack(clean_segments, dim=1)
        denoised_segments = torch.stack(denoised_segments, dim=1)

        if not extended:
            denoised_segments = self.equalize_clip(clean_segments, denoised_segments)
        
        clean_segments = self.normalize(clean_segments, dim=3)
        denoised_segments = self.normalize(denoised_segments, dim=3)

        if extended:
            clean_segments = self.normalize(clean_segments, dim=2)
            denoised_segments = self.normalize(denoised_segments, dim=2)
        
        correlations_components = denoised_segments * clean_segments
        if extended:
            normalization = self.N
        else:
            normalization = self.num_octave_bands
        
        num_segments = torch.maximum((lengths_silent - self.n_fft) // self.hop_length - self.N + 2, torch.zeros_like(lengths_silent, device=lengths_silent.device))
        mask = (torch.arange(correlations_components.shape[1], device=correlations_components.device).unsqueeze(0) < num_segments.unsqueeze(1)).to(correlations_components.dtype)
        
        correlations_components *= mask.unsqueeze(2).unsqueeze(3)
        return torch.sum(correlations_components, dim=(1, 2, 3)) / (normalization*num_segments)
    
    def compute_metric(self, clean_speech: torch.Tensor | None, denoised_speech: torch.Tensor) -> list[dict[str, float]]:
        assert clean_speech is not None
        stois = self.compute_stoi(clean_speech, denoised_speech, extended=False)
        estois = self.compute_stoi(clean_speech, denoised_speech, extended=True)
        return [{"STOI": stoi.item(), "ESTOI": estoi.item()} for stoi, estoi in zip(stois, estois)]
