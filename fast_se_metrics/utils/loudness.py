import torch

# from torchtyping import TensorType
# from typeguard import typechecked

from .bark import centre_of_band_bark_16k, interp

# fmt: off
abs_thresh_power_16k = [
    51286152.000000,     2454709.500000,     70794.593750,     4897.788574,     1174.897705,
    389.045166,     104.712860,     45.708820,     17.782795,     9.772372,
    4.897789,     3.090296,     1.905461,     1.258925,     0.977237,
    0.724436,     0.562341,     0.457088,     0.389045,     0.331131,
    0.295121,     0.269153,     0.257040,     0.251189,     0.251189,
    0.251189,     0.251189,     0.263027,     0.288403,     0.309030,
    0.338844,     0.371535,     0.398107,     0.436516,     0.467735,
    0.489779,     0.501187,     0.501187,     0.512861,     0.524807,
    0.524807,     0.524807,     0.512861,     0.478630,     0.426580,
    0.371535,     0.363078,     0.416869,     0.537032]
# fmt: on

zwicker_power = 0.23
Sl_16k = 1.866055e-001


class Loudness:
    """Apply a loudness curve to the Bark spectrogram

    Parameters
    ----------
    nbark : int
        Number of bark bands

    Attributes
    ----------
    threshs : TensorType[1, 1, "band"]
        Hearing threshold per band; below a band is assumed to contain no significant energy
    exp : TensorType[1, 1, "band"]
        Exponent of each band
    """

    def __init__(self, nbark: int = 49, device: str = "cpu"):
        self.threshs = interp(abs_thresh_power_16k, nbark).unsqueeze(0).unsqueeze(0).to(device)

        exp = 6 / (torch.tensor(centre_of_band_bark_16k) + 2.0)
        self.exp = (exp.clamp(min=1.0, max=2.0) ** 0.15 * zwicker_power).to(device)

    def audible_frame_power(self, bark_bands: torch.Tensor, hearing_threshold_factor: float = 1.0) -> torch.Tensor:
        # Calculate total audible energy for each frame over all bands
        mask = bark_bands > self.threshs * hearing_threshold_factor

        frame_power = torch.sum(bark_bands * mask, dim=2, keepdim=True)
        return frame_power

    def mean_audible_band_power(self, bark_bands: torch.Tensor, frame_is_silent: torch.Tensor) -> torch.Tensor:
        # Calculate mean of audible energy for each band over all frames
        mask = bark_bands > self.threshs * 100.0
        mask = mask * (~frame_is_silent)

        return torch.mean(bark_bands * mask, dim=1)

    def loudness(self, power_density: torch.Tensor) -> torch.Tensor:
        # Calculate audible energy per band
        loudness = (2.0 * self.threshs) ** self.exp * ((0.5 + 0.5 * power_density / self.threshs) ** self.exp - 1)
        loudness[power_density <= self.threshs] = 0.0

        return loudness * Sl_16k
