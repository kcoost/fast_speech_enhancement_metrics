# Fast Speech Enhancement Metrics

A high-performance PyTorch library for computing speech quality metrics with GPU acceleration. Includes optimized implementations of PESQ, STOI, SDR, LSD, DNSMOS, and SpeechBERTScore.

## Installation
Create an environment with Python version 3.12+ and run:
```bash
pip install poetry
poetry install
```

## Usage

```python
import torch
from fast_se_metrics import PESQ, STOI, SDR, LSD, DNSMOS, SpeechBERTScore

# Load your audio (shape: [batch_size, samples])
# 4 samples, 10 second each at 16kHz
clean_speech = torch.randn(4, 160000)
noisy_speech = torch.randn(4, 160000)

# Initialize metrics
pesq = PESQ(sample_rate=16000, use_gpu=True)
stoi = STOI(sample_rate=16000, use_gpu=True)
sdr = SDR(sample_rate=16000, use_gpu=True)

# Compute metrics
pesq_scores = pesq(clean_speech, noisy_speech)
stoi_scores = stoi(clean_speech, noisy_speech)
sdr_scores = sdr(clean_speech, noisy_speech)

print(pesq_scores)  # [{'PESQ': 2.1}, {'PESQ': 1.8}, ...]
print(stoi_scores)  # [{'STOI': 0.85, 'ESTOI': 0.82}, ...]
```

## Performance

Our GPU-accelerated implementations provide significant speedups over existing libraries:

![Performance Comparison](plots/samples_per_second.png)

while maintaining results that are extremely close to the originals

![Performance Comparison](plots/deviations.png)

### Available Metrics

| Metric | Description | Reference | Reference Implementation |
|--------|-------------|------------------|------------------|
| **PESQ** | Perceptual Evaluation of Speech Quality (ITU P.862) | [Berends et al.](https://www.researchgate.net/publication/243773287_Perceptual_evaluation_of_speech_quality_PESQ_-_The_new_ITU_standard_for_end-to-end_speech_quality_assessment_-_Part_II_-_Psychoacoustic_model) | [ludlows](https://github.com/ludlows/PESQ), [AudioLabs](https://github.com/audiolabs/torch-pesq) |
| **(E)STOI** | (Extended) Short-Time Objective Intelligibility | [Taal et al.](https://ieeexplore.ieee.org/document/5713237), [Jensen & Taal](https://ieeexplore.ieee.org/document/7539284) | [mpariente](https://github.com/mpariente/pystoi) |
| **SDR** | Signal-to-Distortion Ratio | [Vincent et al.](https://ieeexplore.ieee.org/abstract/document/1643671), [Scheibler](https://arxiv.org/abs/2110.06440) | [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/audio/signal_distortion_ratio.html) |
| **LSD** | Log-Spectral Distance | [Braun & Tashev](https://arxiv.org/abs/2009.12286) | [Urgent2025](https://github.com/urgent-challenge/urgent2025_challenge/blob/main/evaluation_metrics/calculate_intrusive_se_metrics.py#L66) |
| **DNSMOS** | Deep Noise Suppression Mean Opinion Score | [Reddy et al.](https://arxiv.org/abs/2010.15258) | [DNS-Challenge](https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos_local.py) |
| **SpeechBERTScore** | Semantic similarity using speech embeddings | [Saeki et al.](https://arxiv.org/abs/2401.16812) | [Urgent2025](https://github.com/urgent-challenge/urgent2025_challenge/blob/main/evaluation_metrics/calculate_speechbert_score.py) |

## Benchmarking

To run benchmarks on your system:

```bash
python benchmark_metrics.py
python plot_results.py
```

## Acknowledgments

The PESQ implementation is based on the excellent work by [audiolabs/torch-pesq](https://github.com/audiolabs/torch-pesq).
