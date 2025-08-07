from pathlib import Path
import torch
import torch.nn as nn
from gpu_speech_metrics.base import BaseMetric

INPUT_LENGTH = 9.01
CHECKPOINT_PATH = Path(__file__).parents[1] / "checkpoints" / "SIG_BAK_OVR.pt"


class DNSMOSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_real_stft = nn.Conv1d(320, 161, kernel_size=1, stride=1, bias=False)
        self.conv_imag_stft = nn.Conv1d(320, 161, kernel_size=1, stride=1, bias=False)

        conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv4 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv5 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv6 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv7 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        max_pool = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)

        dense_1 = nn.Linear(64, 128)
        dense_2 = nn.Linear(128, 64)
        dense_3 = nn.Linear(64, 3)

        self.conv_layers = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            conv3,
            nn.ReLU(),
            conv4,
            nn.ReLU(),
            max_pool,
            conv5,
            nn.ReLU(),
            max_pool,
            conv6,
            nn.ReLU(),
            max_pool,
            conv7,
            nn.ReLU(),
        )

        self.output_layers = nn.Sequential(
            dense_1,
            nn.ReLU(),
            dense_2,
            nn.ReLU(),
            dense_3,
        )
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
        self.load_state_dict(checkpoint)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        batch_size = audio.shape[0]

        audio_segments = audio.unfold(1, size=320, step=160)
        audio_segments = audio_segments.transpose(1, 2)

        stft_real = self.conv_real_stft(audio_segments)
        stft_imag = self.conv_imag_stft(audio_segments)

        power_spectrum = stft_real.square() + stft_imag.square()

        power_spectrum = torch.maximum(torch.tensor(1e-12), power_spectrum)
        log_power_spectrum = torch.log10(power_spectrum)
        log_power_spectrum = log_power_spectrum.transpose(1, 2)
        log_power_spectrum = log_power_spectrum.unsqueeze(1)

        hidden = self.conv_layers(log_power_spectrum)

        hidden = hidden.permute(0, 2, 3, 1)
        hidden = hidden.reshape(batch_size, -1, 64)
        hidden = torch.max(hidden, dim=1).values

        output = self.output_layers(hidden)
        return output


class DNSMOS(BaseMetric):
    higher_is_better = True
    EXPECTED_SAMPLING_RATE = 16000

    def __init__(self, sample_rate: int = 16000, use_gpu: bool = False):
        super().__init__(sample_rate, use_gpu)
        self.primary_model = DNSMOSNet()
        self.primary_model.to(self.device)
        self.primary_model.eval()

        try:
            self.primary_model = torch.compile(self.primary_model, mode="default")  # type: ignore[assignment]
        except Exception as e:
            print(f"Warning: Model compilation failed: {e}")

        self.constants = torch.tensor([0.0052439, -0.39604546, 0.04602535], device=self.device)
        self.b1 = torch.tensor([1.22083953, 1.60915514, 1.11546468], device=self.device)
        self.b2 = torch.tensor([-0.08397278, -0.13166888, -0.06766283], device=self.device)

    def compute_metric(
        self, clean_speech: torch.Tensor | None, denoised_speech: torch.Tensor
    ) -> list[dict[str, float]]:
        hop_length = self.EXPECTED_SAMPLING_RATE

        results = []
        # batching doesn't result in any significant speedup
        for audio in denoised_speech:
            while len(audio) < int(INPUT_LENGTH * self.EXPECTED_SAMPLING_RATE):
                audio = torch.cat((audio, audio))

            audio_segments = audio.unfold(0, int(INPUT_LENGTH * hop_length), hop_length)
            audio_segments = audio_segments.float()
            # audio_segments = audio_segments[:7]
            with torch.inference_mode():
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    sig_bak_ovr_raw = self.primary_model(audio_segments)
            sig_bak_ovr = self.constants + self.b1 * sig_bak_ovr_raw + self.b2 * sig_bak_ovr_raw**2
            sig, bak, ovr = sig_bak_ovr.mean(0)
            results.append(
                {
                    "SIG": sig.item(),
                    "BAK": bak.item(),
                    "OVRL": ovr.item(),
                }
            )

            # original has bug:
            # [int(idx * hop_len_samples) - int((idx + INPUT_LENGTH) * hop_len_samples) for idx in range(11)] returns
            # [-144160, -144160, -144160, -144160, -144160, -144160, -144160, -144159, -144159, -144159, -144159]

        return results
