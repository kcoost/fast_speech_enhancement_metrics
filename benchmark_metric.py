import gc
import time
import numpy as np
import torch
from tqdm import tqdm
from gpu_speech_metrics import PESQ, SDR, STOI, LSD, DNSMOS, SpeechBERTScore
from benchmarking.dataloading import load_data
from gpu_speech_metrics.base import BaseMetric
from tests.reference_metrics.LSD_reference import LSD_reference
from tests.reference_metrics.SpeechBERTScore_reference import SpeechBERTScore_reference
from tests.reference_metrics.STOI_reference import STOI_reference
from tests.reference_metrics.PESQ_reference import PESQ_reference_pesq, PESQ_reference_torch_pesq


DEBUG = False
N_BATCHES = 275
BATCH_SIZE = 8
SAMPLE_DURATION = 32
SAMPLE_RATE = 16000
METRIC_CLASSES = [PESQ_reference_pesq, PESQ_reference_torch_pesq, PESQ]



def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def benchmark_metric(metric_class, clean_speech, noisy_speech, use_gpu: bool = False):
    metric = metric_class(sample_rate=SAMPLE_RATE, device="cuda" if use_gpu else "cpu")

    batch_times = []
    values = []
    for idx in tqdm(range(N_BATCHES)):
        start_idx = idx*BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(clean_speech))
        batch_clean = clean_speech[start_idx:end_idx]
        batch_noisy = noisy_speech[start_idx:end_idx]

        start_time = time.time()
        if use_gpu:
            results = metric(batch_clean.cuda(), batch_noisy.cuda())
        else:
            results = metric(batch_clean, batch_noisy)
        end_time = time.time()
        batch_times.append(end_time - start_time)
        #values.extend([np.mean(list(r.values())) for r in results])
        values.extend(results)
        if DEBUG:
            break
    
    if DEBUG:
        return batch_times, values
    else:
        return batch_times[25:], values

if __name__ == "__main__":
    use_milliseconds = False
    if DEBUG:
        clean_speech = torch.load("clean_speech.pt")
        noisy_speech = torch.load("noisy_speech.pt")
    else:
        clean_speech, noisy_speech, _ = load_data(
            sample_duration=SAMPLE_DURATION,
            num_samples=N_BATCHES*BATCH_SIZE,
            sample_rate=SAMPLE_RATE,
            SNR_high=30,
            SNR_low=-15,
        )
    # clean_speech = torch.load("clean_speech.pt")
    # noisy_speech = torch.load("noisy_speech.pt")

    all_results = {}

    for metric_class in METRIC_CLASSES:
        for use_gpu in [True, False]:
            if use_gpu and metric_class.__name__ == "PESQ_reference_pesq":
                continue
            if use_gpu and metric_class.__name__ == "LSD_reference":
                continue
            if "SpeechBERTScore" in metric_class.__name__ and not use_gpu:
                continue
            gc.collect()
            torch.cuda.empty_cache()

            batch_times, values = benchmark_metric(metric_class=metric_class, clean_speech=clean_speech, noisy_speech=noisy_speech, use_gpu=use_gpu)
            print(f"\n{'='*50}")
            print(f"{metric_class.__name__} Benchmark Results ({'GPU' if use_gpu else 'CPU'})")
            #print(f"Number of batches: {len(batch_times)}")
            if use_milliseconds:
                print(f"Average time per batch: {np.mean(batch_times)*1000:.4f} ± {np.std(batch_times)*1000:.4f} milliseconds")
            else:
                print(f"Average time per batch: {np.mean(batch_times):.4f} ± {np.std(batch_times):.4f} seconds")

            all_results[metric_class.__name__ + ("_GPU" if use_gpu else "_CPU")] = {
                "batch_times": batch_times,
                "values": values
            }
    import json
    import os

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    all_results["batch_size"] = BATCH_SIZE
    all_results["sample_duration"] = SAMPLE_DURATION
    all_results["sample_rate"] = SAMPLE_RATE
    all_results["SNR_high"] = 30
    all_results["SNR_low"] = -15
    with open(f"results/{metric_class.__name__}_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Calculate average absolute deviation between all metric results
    metric_names = list(all_results.keys())
    for i in range(len(metric_names)):
        for j in range(i + 1, len(metric_names)):
            metric1 = metric_names[i]
            metric2 = metric_names[j]
            values1 = all_results[metric1]["values"]
            values2 = all_results[metric2]["values"]
            mean_abs_dev = np.mean(np.abs(np.array(values1) - np.array(values2)))
            print(f"Mean absolute deviation between {metric1} and {metric2}: {mean_abs_dev:.4f}")
