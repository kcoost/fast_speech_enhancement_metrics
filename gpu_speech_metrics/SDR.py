import math
import torch
from torch.linalg import norm
from gpu_speech_metrics.base import BaseMetric


def _symmetric_toeplitz_solve(r_0: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fast Toeplitz system solver with multiple strategies."""
    # For now, use the optimized Toeplitz construction with standard solver
    # This is more reliable than our current Levinson-Durbin implementation

    n = r_0.shape[-1]

    # Create Toeplitz matrix more efficiently using broadcasting
    indices = torch.arange(n, device=r_0.device)
    row_indices = indices.unsqueeze(0)
    col_indices = indices.unsqueeze(1)
    toeplitz_indices = torch.abs(row_indices - col_indices)

    # Broadcast to create Toeplitz matrix
    r_matrix = r_0[..., toeplitz_indices]

    # Use Cholesky decomposition for symmetric positive definite matrix
    try:
        L = torch.linalg.cholesky(r_matrix)
        y = torch.linalg.solve_triangular(L, b.unsqueeze(-1), upper=False)
        sol = torch.linalg.solve_triangular(L.transpose(-2, -1), y, upper=True)
        return sol.squeeze(-1)
    except Exception:
        # Fallback to regular solver if Cholesky fails
        return torch.linalg.solve(r_matrix, b)


def _compute_autocorr_crosscorr(
    target: torch.Tensor, preds: torch.Tensor, corr_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # the valid length for the signal after convolution
    n_fft = 2 ** math.ceil(math.log2(preds.shape[-1] + target.shape[-1] - 1))

    # computes the auto correlation of `target`
    # r_0 is the first row of the symmetric Toeplitz metric
    t_fft = torch.fft.rfft(target, n=n_fft, dim=-1)
    r_0 = torch.fft.irfft(torch.abs(t_fft) ** 2, n=n_fft)[..., :corr_len]

    # computes the cross-correlation of `target` and `preds`
    p_fft = torch.fft.rfft(preds, n=n_fft, dim=-1)
    b = torch.fft.irfft(t_fft.conj() * p_fft, n=n_fft, dim=-1)[..., :corr_len]

    return r_0, b


class SDR(BaseMetric):
    higher_is_better = True
    EXPECTED_SAMPLING_RATE = 16000

    def __init__(self, sample_rate: int = 16000, use_gpu: bool = False):
        super().__init__(sample_rate, use_gpu)
        self.filter_length = 512
        self.zero_mean = False
        self.load_diag = None

    def preprocess_speech(self, speech: torch.Tensor) -> torch.Tensor:
        speech = speech.to(torch.float32)
        if self.zero_mean:
            speech = speech - speech.mean(dim=-1, keepdim=True)

        speech_norm = torch.clamp(norm(speech, dim=-1, keepdim=True), min=1e-6)
        speech = speech / speech_norm
        return speech

    def compute_metric(
        self, clean_speech: torch.Tensor | None, denoised_speech: torch.Tensor
    ) -> list[dict[str, float]]:
        assert clean_speech is not None

        clean_speech = self.preprocess_speech(clean_speech)
        denoised_speech = self.preprocess_speech(denoised_speech)

        # Optimized correlation computation
        r_0, b = _compute_autocorr_crosscorr(clean_speech, denoised_speech, corr_len=self.filter_length)
        r_0 = r_0.to(torch.float32)
        b = b.to(torch.float32)

        if self.load_diag is not None:
            r_0[..., 0] += self.load_diag

        # Fast Toeplitz system solver
        sol = _symmetric_toeplitz_solve(r_0, b)

        # Optimized coherence computation using einsum
        coh = torch.einsum("...l,...l->...", b, sol)

        # Compute ratio and convert to dB with numerical stability
        ratio = coh / torch.clamp(1 - coh, min=1e-8)
        val = 10.0 * torch.log10(torch.clamp(ratio, min=1e-8))

        return [{"SDR": sdr.item()} for sdr in val]
