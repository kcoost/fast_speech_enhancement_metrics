import gc
from pathlib import Path
import time
import torch
import json
import os
from tqdm import tqdm
from benchmarking.dataloading import load_audio_data
from fast_se_metrics import PESQ, SDR, STOI, LSD, DNSMOS, SpeechBERTScore
from tests.reference_metrics.PESQ_reference import PESQ_reference_pesq, PESQ_reference_torch_pesq
from tests.reference_metrics.SDR_reference import SDR_reference
from tests.reference_metrics.STOI_reference import STOI_reference
from tests.reference_metrics.LSD_reference import LSD_reference
from tests.reference_metrics.DNSMOS_reference import DNSMOS_reference
from tests.reference_metrics.SpeechBERTScore_reference import SpeechBERTScore_reference

N_SAMPLES = 8192
CUTOFF_FRACTION = 0.15
SAMPLE_DURATION = 16
SAMPLE_RATE = 16000
METRIC_CLASSES = []
METRIC_CLASSES.append([PESQ_reference_pesq, PESQ_reference_torch_pesq, PESQ])
METRIC_CLASSES.append([SDR_reference, SDR])
METRIC_CLASSES.append([STOI_reference, STOI])
METRIC_CLASSES.append([LSD_reference, LSD])
METRIC_CLASSES.append([DNSMOS_reference, DNSMOS])  # type: ignore[list-item]
METRIC_CLASSES.append([SpeechBERTScore_reference, SpeechBERTScore])  # type: ignore[list-item]


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def iterate_over_metrics(metric_group):
    for metric_class in metric_group:
        for use_gpu in [True, False]:
            if use_gpu and metric_class.__name__ in ["PESQ_reference_pesq", "STOI_reference", "LSD_reference"]:
                continue
            if "SpeechBERTScore" in metric_class.__name__ and not use_gpu:
                continue
            if "DNSMOS" in metric_class.__name__ and not use_gpu:
                continue
            yield metric_class, use_gpu


def benchmark_metric(metric, clean_speech, noisy_speech, batch_size):
    print(f"{'='*75}")
    print(f"Benchmarking {metric.__class__.__name__} on {metric.device} with batch size {batch_size}")
    print(f"{'='*75}")
    batch_times = []
    values = []
    n_iterations = N_SAMPLES // batch_size

    # PESQ is very slow
    if "PESQ" in metric.__class__.__name__:
        n_iterations = min(n_iterations, 128)
    else:
        n_iterations = min(n_iterations, 512)
    for idx in tqdm(range(n_iterations), mininterval=5):
        start_idx = idx * batch_size
        end_idx = min(start_idx + batch_size, len(clean_speech))
        batch_clean = clean_speech[start_idx:end_idx]
        batch_noisy = noisy_speech[start_idx:end_idx]

        batch_clean = batch_clean.to(metric.device)
        batch_noisy = batch_noisy.to(metric.device)

        # memory_before = torch.cuda.memory_allocated(0) / (1024 ** 2)
        start_time = time.time()
        results = metric(batch_clean, batch_noisy)
        end_time = time.time()
        batch_times.append(end_time - start_time)
        values.extend(results)

    del metric
    clean_memory()

    return {
        "batch_times": batch_times[int(len(batch_times) * CUTOFF_FRACTION + 1) :],
        "values": values,
    }


def benchmark_metric_group(metric_group, clean_speech, noisy_speech, snrs, batch_size):
    all_results = {}

    for metric_class, use_gpu in iterate_over_metrics(metric_group):
        clean_memory()
        metric = metric_class(sample_rate=SAMPLE_RATE, use_gpu=use_gpu)
        results = benchmark_metric(metric, clean_speech, noisy_speech, batch_size)
        all_results[metric_class.__name__ + ("_GPU" if use_gpu else "_CPU")] = results
        time.sleep(20)

    os.makedirs(Path(__file__).parent / "results" / f"batch_size_{batch_size}", exist_ok=True)
    all_results["snrs"] = snrs.squeeze().tolist()
    all_results["batch_size"] = batch_size
    all_results["sample_duration"] = SAMPLE_DURATION
    all_results["sample_rate"] = SAMPLE_RATE
    all_results["SNR_high"] = 25
    all_results["SNR_low"] = -5
    with open(
        Path(__file__).parent / "results" / f"batch_size_{batch_size}" / f"{metric_group[-1].__name__}_results.json",
        "w",
    ) as f:
        json.dump(all_results, f, indent=4)


def main(clean_speech, noisy_speech, snrs, batch_size):
    for metric_group in METRIC_CLASSES:
        benchmark_metric_group(
            metric_group=metric_group,
            clean_speech=clean_speech,
            noisy_speech=noisy_speech,
            snrs=snrs,
            batch_size=batch_size,
        )


if __name__ == "__main__":
    clean_speech, noisy_speech, snrs = load_audio_data(
        sample_duration=SAMPLE_DURATION,
        num_samples=N_SAMPLES,
        sample_rate=SAMPLE_RATE,
    )
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        main(clean_speech, noisy_speech, snrs, batch_size)
