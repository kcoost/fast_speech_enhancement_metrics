# code copied from https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/dnsmos.py

import pytest
import subprocess
from onnx2torch import convert

import torch
import numpy as np
import torchaudio
from gpu_speech_metrics import DNSMOS

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


def poly1d(coefficients, use_numpy=False):
    if use_numpy:
        return np.poly1d(coefficients)
    coefficients = tuple(reversed(coefficients))

    def func(p):
        return sum(coef * p**i for i, coef in enumerate(coefficients))

    return func

class DNSMOS_local:
    # ported from
    # https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos_local.py
    def __init__(self, primary_model_path, use_gpu=False):
        self.use_gpu = use_gpu

        if primary_model_path is not None:
            self.primary_model = convert(primary_model_path).eval()
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=321, hop_length=160, pad_mode="constant"
        )

        self.to_db = torchaudio.transforms.AmplitudeToDB("power", top_db=80.0)
        if use_gpu:
            if primary_model_path is not None:
                self.primary_model = self.primary_model.cuda()
                self.p808_model = self.p808_model.cuda()
            self.spectrogram = self.spectrogram.cuda()

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        flag = False
        if is_personalized_MOS:
            p_ovr = poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046], flag)
            p_sig = poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726], flag)
            p_bak = poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132], flag)
        else:
            p_ovr = poly1d([-0.06766283, 1.11546468, 0.04602535], flag)
            p_sig = poly1d([-0.08397278, 1.22083953, 0.0052439], flag)
            p_bak = poly1d([-0.13166888, 1.60915514, -0.39604546], flag)

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, aud, input_fs, is_personalized_MOS=False):
        device = "cuda" if self.use_gpu else "cpu"
        if isinstance(aud, torch.Tensor):
            aud = aud.to(device=device)
        else:
            aud = torch.as_tensor(aud, dtype=torch.float32, device=device)
        
        assert input_fs == SAMPLING_RATE
        audio = aud
        len_samples = int(INPUT_LENGTH * SAMPLING_RATE)
        while len(audio) < len_samples:
            audio = torch.cat((audio, audio))

        num_hops = int(np.floor(len(audio) / SAMPLING_RATE) - INPUT_LENGTH) + 1
        hop_len_samples = SAMPLING_RATE
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)
            ]
            if len(audio_seg) < len_samples:
                continue

            input_features = audio_seg.float()[None, :]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.primary_model(
                input_features
            )[0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS
            )
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)

        to_array = torch.stack
        return {
            "OVRL_raw": to_array(predicted_mos_ovr_seg_raw).mean(),
            "SIG_raw": to_array(predicted_mos_sig_seg_raw).mean(),
            "BAK_raw": to_array(predicted_mos_bak_seg_raw).mean(),
            "OVRL": to_array(predicted_mos_ovr_seg).mean(),
            "SIG": to_array(predicted_mos_sig_seg).mean(),
            "BAK": to_array(predicted_mos_bak_seg).mean(),
        }

def test_dnsmos(speech_data, tmp_path):
    noisy_speeches = speech_data["noisy_speech"]

    model_path = tmp_path / "sig_bak_ovr.onnx"
    subprocess.run([
            "wget", "-c", "-O", model_path,
            "https://github.com/microsoft/DNS-Challenge/raw/refs/heads/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
        ], check=True)
    dnsmos_reference = DNSMOS_local(primary_model_path=model_path, use_gpu=False)
    dnsmos = DNSMOS()

    reference_results = []
    for noisy_speech in noisy_speeches:
        with torch.no_grad():
            reference_result = dnsmos_reference(noisy_speech, 16000)
            reference_results.append(reference_result)
    results = dnsmos(None, noisy_speeches)

    for reference_result, result in zip(reference_results, results):
        assert reference_result["OVRL"] == pytest.approx(result["OVRL"], 1e-6)
        assert reference_result["SIG"] == pytest.approx(result["SIG"], 1e-6)
        assert reference_result["BAK"] == pytest.approx(result["BAK"], 1e-6)
