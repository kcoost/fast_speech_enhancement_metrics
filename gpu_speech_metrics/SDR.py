import math
import torch
from torch.linalg import norm
from gpu_speech_metrics.base import BaseMetric
# From https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/audio/sdr.py#L88-L197

def _symmetric_toeplitz(vector: torch.Tensor) -> torch.Tensor:
    """Construct a symmetric Toeplitz matrix using one vector.

    Args:
        vector: shape [..., L]

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.functional.audio.sdr import _symmetric_toeplitz
        >>> v = tensor([0, 1, 2, 3, 4])
        >>> _symmetric_toeplitz(v)
        tensor([[0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0]])

    Returns:
        a symmetric Toeplitz matrix of shape [..., L, L]

    """
    vec_exp = torch.cat([torch.flip(vector, dims=(-1,)), vector[..., 1:]], dim=-1)
    v_len = vector.shape[-1]
    return torch.as_strided(
        vec_exp, size=vec_exp.shape[:-1] + (v_len, v_len), stride=vec_exp.stride()[:-1] + (1, 1)
    ).flip(dims=(-1,))

def _compute_autocorr_crosscorr(target: torch.Tensor, preds: torch.Tensor, corr_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the auto correlation of `target` and the cross correlation of `target` and `preds`.

    This calculation is done using the fast Fourier transform (FFT). Let's denotes the symmetric Toeplitz metric of the
    auto correlation of `target` as `R`, the cross correlation as 'b', then solving the equation `Rh=b` could have `h`
    as the coordinate of `preds` in the column space of the `corr_len` shifts of `target`.

    Args:
        target: the target (reference) signal of shape [..., time]
        preds: the preds (estimated) signal of shape [..., time]
        corr_len: the length of the auto correlation and cross correlation

    Returns:
        the auto correlation of `target` of shape [..., corr_len]
        the cross correlation of `target` and `preds` of shape [..., corr_len]

    """
    # the valid length for the signal after convolution
    n_fft = 2 ** math.ceil(math.log2(preds.shape[-1] + target.shape[-1] - 1))

    # computes the auto correlation of `target`
    # r_0 is the first row of the symmetric Toeplitz metric
    t_fft = torch.fft.rfft(target, n=n_fft, dim=-1)
    r_0 = torch.fft.irfft(t_fft.real**2 + t_fft.imag**2, n=n_fft)[..., :corr_len]

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

    def compute_metric(self, clean_speech: torch.Tensor, denoised_speech: torch.Tensor) -> list[dict[str, float]]:
        # use double precision
        clean_speech_dtype = clean_speech.dtype
        clean_speech = clean_speech.double()
        denoised_speech = denoised_speech.double()

        if self.zero_mean:
            clean_speech = clean_speech - clean_speech.mean(dim=-1, keepdim=True)
            denoised_speech = denoised_speech - denoised_speech.mean(dim=-1, keepdim=True)

        # normalize along time-axis to make clean_speech and denoised_speech have unit norm
        denoised_speech = denoised_speech / torch.clamp(norm(denoised_speech, dim=-1, keepdim=True), min=1e-6)
        clean_speech = clean_speech / torch.clamp(norm(clean_speech, dim=-1, keepdim=True), min=1e-6)

        # solve for the optimal filter
        # compute auto-correlation and cross-correlation
        r_0, b = _compute_autocorr_crosscorr(denoised_speech, clean_speech, corr_len=self.filter_length)

        if self.load_diag is not None:
            # the diagonal factor of the Toeplitz matrix is the first coefficient of r_0
            r_0[..., 0] += self.load_diag

        # regular matrix solver
        r = _symmetric_toeplitz(r_0)  # the auto-correlation of the L shifts of `denoised_speech`
        sol = torch.linalg.solve(r, b)

        # compute the coherence
        coh = torch.einsum("...l,...l->...", b, sol)

        # transform to decibels
        ratio = coh / (1 - coh)
        val = 10.0 * torch.log10(ratio)

        if clean_speech_dtype == torch.float64:
            return val
        return [{"SDR": sdr.item()} for sdr in val]


