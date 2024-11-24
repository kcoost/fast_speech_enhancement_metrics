import numpy as np
import warnings

import numpy as np
import torch



# class STOI:
#     def __init__(self):
#         self.sampling_frequency = 10000
#         self.win_length = 256
#         self.hop_length = self.win_length // 2
#         self.n_fft = 512
#         self.num_octave_bands = 15
#         self.min_frequency = 150 # Center frequency of 1st octave band (Hz)
#         self.octave_band_matrix = self.get_octave_band_matrix()
#         self.N = 30 # N. frames for intermediate intelligibility
#         self.beta = -15. # Lower SDR bound
#         self.dynamic_range = 40
    
#     def get_octave_band_matrix(self) -> torch.Tensor:
#         # Computes the 1/3 octave band matrix
#         num_frequencies = self.n_fft//2 + 1
        
#         frequencies = torch.linspace(0, self.sampling_frequency//2, num_frequencies, dtype=torch.float64)
        
#         # Calculate center frequencies and frequency bounds
#         band_idx = torch.arange(self.num_octave_bands, dtype=torch.float64) # important for precision
#         frequencies_low = self.min_frequency * torch.pow(2.0, (2 * band_idx - 1) / 6)
#         frequencies_high = self.min_frequency * torch.pow(2.0, (2 * band_idx + 1) / 6)
        
#         octave_band_matrix = torch.zeros((self.num_octave_bands, num_frequencies), dtype=torch.float64)        
#         for i in range(self.num_octave_bands):
#             # Match 1/3 oct band freq with fft frequency bin
#             idx_bin_low = torch.argmin((frequencies - frequencies_low[i]).abs())
#             idx_bin_high = torch.argmin((frequencies - frequencies_high[i]).abs())

#             frequencies_low[i] = frequencies[idx_bin_low]
#             frequencies_high[i] = frequencies[idx_bin_high]
            
#             octave_band_matrix[i, idx_bin_low:idx_bin_high] = 1
        
#         return octave_band_matrix.to(torch.float32)


#     def stft(self, speech: torch.Tensor) -> torch.Tensor:
#         window = torch.hann_window(self.win_length+1, dtype=speech.dtype)[1:]

#         spectrogram = torch.stft(
#             speech,
#             n_fft=self.n_fft,
#             hop_length=self.hop_length,
#             win_length=self.win_length,
#             window=window,
#             center=False,
#             normalized=False,
#             return_complex=True,
#             onesided=True
#         )
#         return spectrogram.abs().square()

#     def overlap_and_add(self, frames: torch.Tensor) -> torch.Tensor:
#         num_frames, frame_length = frames.shape
#         final_length = (num_frames - 1) * self.hop_length + frame_length
        
#         signal = torch.zeros(final_length, dtype=frames.dtype, device=frames.device)

#         # Vectorized version of
#         # for i in range(num_frames):
#         #     signal[i * hop_length:i * hop_length + frame_length] += x_frames[i]
#         idx = torch.arange(frame_length).unsqueeze(0) + self.hop_length * torch.arange(num_frames).unsqueeze(1)
#         signal = signal.scatter_add(0, idx.flatten(), frames.flatten())
#         return signal

#     def remove_silent_frames(self, clean_speech: torch.Tensor, denoised_speech: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         # Compute Mask
#         window = torch.hann_window(self.win_length + 1, dtype=clean_speech.dtype)[1:]

#         # Create frames
#         clean_frames = torch.stack([
#             window * clean_speech[i:i + self.win_length] for i in range(0, len(clean_speech) - self.win_length, self.hop_length)
#         ])
#         denoised_frames = torch.stack([
#             window * denoised_speech[i:i + self.win_length] for i in range(0, len(denoised_speech) - self.win_length, self.hop_length)
#         ])

#         # Compute energies in dB
#         clean_energies = 20 * torch.log10(torch.norm(clean_frames, dim=1) + 1e-9)

#         # Find boolean mask of energies lower than dynamic_range dB
#         # with respect to maximum clean speech energy frame
#         mask = (torch.max(clean_energies) - self.dynamic_range - clean_energies) < 0

#         # Remove silent frames by masking
#         clean_frames = clean_frames[mask]
#         denoised_frames = denoised_frames[mask]

#         clean_silent = self.overlap_and_add(clean_frames)
#         denoised_silent = self.overlap_and_add(denoised_frames)

#         return clean_silent, denoised_silent
    
#     def normalize(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
#         x -= x.mean(dim=dim, keepdim=True)
#         norm = torch.norm(x, p=2, dim=dim, keepdim=True)
#         x /= norm
#         return x

#     def row_col_normalize(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.normalize(x, dim=2)
#         x = self.normalize(x, dim=1)
#         return x
    
#     def compute_segments(self, speech: torch.Tensor):
#         spectrogram = self.stft(speech)
#         tob = torch.sqrt(torch.matmul(self.octave_band_matrix, spectrogram))
#         segments = [tob[:, m:m + self.N] for m in range(max(tob.shape[1] - self.N + 1, 0))]
#         return segments

#     def __call__(self, clean_speech: torch.Tensor, denoised_speech: torch.Tensor, extended: bool = False):
#         if clean_speech.shape != denoised_speech.shape:
#             raise Exception('`clean_speech` and `denoised_speech` should have the same shape.')

#         clean_speech, denoised_speech = self.remove_silent_frames(clean_speech, denoised_speech)

#         clean_segments = self.compute_segments(clean_speech)
#         denoised_segments = self.compute_segments(denoised_speech)

#         # Ensure at least 30 frames for intermediate intelligibility
#         if len(clean_segments) == 0:
#             warnings.warn('Not enough non-silent frames. Please check your sound files', RuntimeWarning)
#             return 0
        
#         clean_segments = torch.stack(clean_segments)
#         denoised_segments = torch.stack(denoised_segments)

#         if extended:
#             clean_segments_normalized = self.row_col_normalize(clean_segments)
#             denoised_segments_normalized = self.row_col_normalize(denoised_segments)
#             correlations_components = clean_segments_normalized * denoised_segments_normalized
#             return torch.mean(correlations_components, dim=(0, 2)).sum()
#         else:
#             # Find normalization constants and normalize
#             normalization_consts = torch.norm(clean_segments, dim=2, keepdim=True) /(torch.norm(denoised_segments, dim=2, keepdim=True) + 1e-9)
#             denoised_segments_normalized = denoised_segments * normalization_consts

#             # Clip as described in [1]
#             clip_value = 10 ** (-self.beta / 20)
#             denoised_segments_normalized = torch.minimum(
#                 denoised_segments_normalized, clean_segments * (1 + clip_value))

#             denoised_segments_normalized = self.normalize(denoised_segments_normalized, dim=2)
#             clean_segments = self.normalize(clean_segments, dim=2)

#             correlations_components = denoised_segments_normalized * clean_segments
#             return torch.mean(correlations_components, dim=(0, 1)).sum()

class STOI:
    def __init__(self):
        self.sampling_frequency = 10000
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

    def overlap_and_add(self, frames: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        final_lengths = (lengths + 1) * self.hop_length
        max_length = torch.max(final_lengths)
        batch_size = len(final_lengths)
        
        signal = torch.zeros((batch_size, max_length.item()), dtype=frames.dtype, device=frames.device)

        for i, frame in enumerate(frames.split(lengths.tolist())):
            # Vectorized version of
            # for i in range(num_frames):
            #     signal[i * hop_length:i * hop_length + frame_length] += x_frames[i]
            idx = torch.arange(self.win_length, device=frames.device).unsqueeze(0) + self.hop_length * torch.arange(lengths[i], device=frames.device).unsqueeze(1)
            signal[i] += signal[i].scatter_add(0, idx.flatten(), frame.flatten())
        return signal, final_lengths

    def remove_silent_frames(self, clean_speech: torch.Tensor, denoised_speech: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    def __call__(self, clean_speech: torch.Tensor, denoised_speech: torch.Tensor, extended: bool = False):
        if clean_speech.shape != denoised_speech.shape:
            raise Exception('`clean_speech` and `denoised_speech` should have the same shape.')
        
        clean_speech = torch.atleast_2d(clean_speech)
        denoised_speech = torch.atleast_2d(denoised_speech)

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

if __name__ == "__main__":
    # Test STOI calculation
    # fs = 10000
    # torch.manual_seed(1)
    # x1 = torch.randn(fs)
    # x2 = torch.randn(fs)
    # x2[1000:3000] *= 0.01
    # y1 = x1 + 0.1 * torch.randn(fs)
    # y2 = x2 + 0.3 * torch.randn(fs)

    # torch_stoi = STOI()
    
    # d = torch_stoi(torch.stack([x1, x2]).cuda(), torch.stack([y1, y2]).cuda(), extended=True)
    # from pystoi import stoi
    # d_ref1 = stoi(x1.numpy(), y1.numpy(), fs, extended=True)
    # d_ref2 = stoi(x2.numpy(), y2.numpy(), fs, extended=True)
    # print(d[0].item() - d_ref1, d[1].item() - d_ref2)

    fs = 10000
    num_batches = 10
    batch_size = 32
    
    torch.manual_seed(1)
    x1 = torch.randn(num_batches*batch_size, 64*fs)
    y1 = x1 + 0.25 * torch.randn_like(x1)

    x1_numpy = x1.numpy()
    y1_numpy = y1.numpy()

    import time
    from pystoi import stoi

    start = time.time()
    for i in range(num_batches*batch_size):
        d = stoi(x1_numpy[i], y1_numpy[i], fs, extended=True)
    end = time.time()
    print(f"Time taken for pystoi: {end - start} seconds")

    start = time.time()
    torch_stoi = STOI()
    for x1_batch, y1_batch in zip(x1.split(batch_size), y1.split(batch_size)):
        d = torch_stoi(x1_batch.cuda(), y1_batch.cuda(), extended=True)
    end = time.time()
    print(f"Time taken for torch_stoi: {end - start} seconds")
