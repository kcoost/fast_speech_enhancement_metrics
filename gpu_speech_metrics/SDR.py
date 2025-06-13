import math
import torch
from torch.linalg import norm
from gpu_speech_metrics.base import BaseMetric

def _symmetric_toeplitz_solve(r_0: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fast Toeplitz system solver with multiple strategies."""
    # For now, use the optimized Toeplitz construction with standard solver
    # This is more reliable than our current Levinson-Durbin implementation
    
    batch_shape = r_0.shape[:-1]
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
    except:
        # Fallback to regular solver if Cholesky fails
        return torch.linalg.solve(r_matrix, b)

def _compute_autocorr_crosscorr(target: torch.Tensor, preds: torch.Tensor, corr_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    # the valid length for the signal after convolution
    n_fft = 2 ** math.ceil(math.log2(preds.shape[-1] + target.shape[-1] - 1))

    # computes the auto correlation of `target`
    # r_0 is the first row of the symmetric Toeplitz metric
    t_fft = torch.fft.rfft(target, n=n_fft, dim=-1)
    r_0 = torch.fft.irfft(torch.abs(t_fft)**2, n=n_fft)[..., :corr_len]

    # computes the cross-correlation of `target` and `preds`
    p_fft = torch.fft.rfft(preds, n=n_fft, dim=-1)
    b = torch.fft.irfft(t_fft.conj() * p_fft, n=n_fft, dim=-1)[..., :corr_len]

    return r_0, b

class SDR(BaseMetric):
    higher_is_better = True
    EXPECTED_SAMPLING_RATE = 16000

    def __init__(self, sample_rate: int = 16000, device: str = "cpu"):
        super().__init__(sample_rate, device)
        self.filter_length = 512
        self.zero_mean = False
        self.load_diag = None

    def compute_metric(self, clean_speech: torch.Tensor | None, denoised_speech: torch.Tensor) -> list[dict[str, float]]:
        assert clean_speech is not None

        clean_speech = clean_speech.to(torch.float32)
        denoised_speech = denoised_speech.to(torch.float32)

        if self.zero_mean:
            clean_speech = clean_speech - clean_speech.mean(dim=-1, keepdim=True)
            denoised_speech = denoised_speech - denoised_speech.mean(dim=-1, keepdim=True)

        # Normalize with optimized operations
        denoised_norm = torch.clamp(norm(denoised_speech, dim=-1, keepdim=True), min=1e-6)
        clean_norm = torch.clamp(norm(clean_speech, dim=-1, keepdim=True), min=1e-6)
        
        denoised_speech = denoised_speech / denoised_norm
        clean_speech = clean_speech / clean_norm

        # Optimized correlation computation
        r_0, b = _compute_autocorr_crosscorr(denoised_speech, clean_speech, 
                                                      corr_len=self.filter_length)

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