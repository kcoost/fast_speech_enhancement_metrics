import time
import pandas as pd
import torch
from tqdm import tqdm
import cpuinfo
from pystoi import stoi
from datasets import load_dataset
from torchaudio.functional import resample
from src.gpu_speech_metrics.stoi import STOI
import seaborn as sns
import matplotlib.pyplot as plt
torch.manual_seed(42)
CACHE = False

def load_noise(duration: int = 1):
    flap_noise = load_dataset("nccratliri/wing-flap-noise-audio-examples", streaming=not CACHE)
    noises = torch.tensor([])
    for sample in tqdm(flap_noise["train"], desc="Loading noise"):
        noise = resample(torch.tensor(sample["audio"]["array"]),
                        orig_freq=sample["audio"]["sampling_rate"],
                        new_freq=10000)
        noises = torch.cat([noises, noise])
        if len(noises) >= duration * 10000:
            noises = noises[:duration * 10000]
            return noises
    
    noises = torch.cat(((duration * 10000) //len(noises) + 1)*[noises])
    noises = noises[:duration * 10000]
    return noises

def load_speech(duration: int = 1):
    librispeech = load_dataset("openslr/librispeech_asr", streaming=not CACHE)
    speeches = torch.tensor([])
    for sample in tqdm(librispeech["train.other.500"], desc="Loading speech"):
        speech = resample(torch.tensor(sample["audio"]["array"]),
                      orig_freq=sample["audio"]["sampling_rate"],
                      new_freq=10000)
        speeches = torch.cat([speeches, speech])
        if len(speeches) >= duration * 10000:
            speeches = speeches[:duration * 10000]
            return speeches
    speeches = torch.cat(((duration * 10000) //len(speeches) + 1)*[speeches])
    speeches = speeches[:duration * 10000]
    return speeches

def load_data(sample_duration: int = 1, num_samples: int = 1):
    speech = load_speech(num_samples * sample_duration)
    noise = load_noise(num_samples * sample_duration)
    
    speech = speech.view(num_samples, sample_duration * 10000)
    noise = noise.view(num_samples, sample_duration * 10000)

    speech_rms = (speech.square()).mean(dim=1, keepdim=True).sqrt()
    noise_rms = (noise.square()).mean(dim=1, keepdim=True).sqrt()
    SNR_high = 15
    SNR_low = -10

    snr = torch.rand(num_samples, 1) * (SNR_high - SNR_low) + SNR_low
    noise_scale = speech_rms / (10 ** (snr / 20)) / (noise_rms + 1e-12)
    noisy_speech = speech + noise_scale * noise

    return speech.float(), noisy_speech.float(), snr

if __name__ == "__main__":
    NUM_SAMPLES = 480
    BATCH_SIZE = 32
    SAMPLE_DURATION = 32
    speech, noisy_speech, snr = load_data(SAMPLE_DURATION, NUM_SAMPLES)

    torch_stoi = STOI()

    torch_stoi_cpu_results = []
    start = time.time()
    for speech_batch, noisy_speech_batch in tqdm(zip(speech.split(BATCH_SIZE), noisy_speech.split(BATCH_SIZE)), desc="torch_stoi (CPU)"):
        d = torch_stoi(speech_batch, noisy_speech_batch, extended=True)
        torch_stoi_cpu_results.append(d)
    end = time.time()
    speed_torch_stoi_cpu = (end - start) / NUM_SAMPLES
    print(f"Time taken for torch_stoi (CPU): {speed_torch_stoi_cpu:.4f} seconds/sample")
    
    start = time.time()
    torch_stoi_gpu_results = []
    for speech_batch, noisy_speech_batch in tqdm(zip(speech.split(BATCH_SIZE), noisy_speech.split(BATCH_SIZE)), desc="torch_stoi (GPU)"):
        d = torch_stoi(speech_batch.cuda(), noisy_speech_batch.cuda(), extended=True)
        torch_stoi_gpu_results.append(d.cpu())
    end = time.time()
    speed_torch_stoi_gpu = (end - start) / NUM_SAMPLES
    print(f"Time taken for torch_stoi (GPU): {speed_torch_stoi_gpu:.4f} seconds/sample")

    speech_numpy = speech.numpy()
    noisy_speech_numpy = noisy_speech.numpy()
    pystoi_results = []
    start = time.time()
    for i in tqdm(range(NUM_SAMPLES), desc="pystoi"):
        d = stoi(speech_numpy[i], noisy_speech_numpy[i], 10000, extended=True)
        pystoi_results.append(d)
    end = time.time()
    speed_pystoi = (end - start) / NUM_SAMPLES
    print(f"Time taken for pystoi: {speed_pystoi:.4f} seconds/sample")

    data = pd.DataFrame({
        "Method": ["torch_stoi_cpu", "torch_stoi_gpu", "pystoi"],
        "samples/second": [1/speed_torch_stoi_cpu, 1/speed_torch_stoi_gpu, 1/speed_pystoi]
    })
    plt.figure(figsize=(7, 6))
    sns.barplot(data, x="Method", y="samples/second", hue="Method", errorbar=None)
    plt.title(f"Benchmarking STOI methods\nCPU: {cpuinfo.get_cpu_info()['brand_raw']}, GPU: {torch.cuda.get_device_name()}\nSample duration: {SAMPLE_DURATION} seconds, Batch size: {BATCH_SIZE}")
    plt.xlabel("")
    plt.savefig(f"benchmark_methods_duration={SAMPLE_DURATION}_batch={BATCH_SIZE}.png")

    plt.yscale("log")
    plt.savefig(f"benchmark_methods_duration={SAMPLE_DURATION}_batch={BATCH_SIZE}_log.png")
    plt.clf()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Create paired dataframes for each comparison
    cpu_pystoi_df = pd.DataFrame({
        'torch_stoi_cpu': sum([r.tolist() for r in torch_stoi_cpu_results], []),
        'pystoi': [float(r) for r in pystoi_results],
        'SNR': snr[:, 0].tolist()
    })

    gpu_pystoi_df = pd.DataFrame({
        'torch_stoi_gpu': sum([r.tolist() for r in torch_stoi_gpu_results], []),
        'pystoi': [float(r) for r in pystoi_results],
        'SNR': snr[:, 0].tolist()
    })

    # Plot using seaborn
    # Set background color to light grey
    axes[0].set_facecolor('#f0f0f0')
    axes[1].set_facecolor('#f0f0f0')
    
    axes[0].plot([0, 1], [0, 1], '--', color=sns.color_palette()[0])  # Add diagonal reference line
    sns.scatterplot(data=cpu_pystoi_df, x='torch_stoi_cpu', y='pystoi', hue='SNR', palette="magma", ax=axes[0])
    axes[0].set_title("torch_stoi (CPU) vs pystoi")

    axes[1].plot([0, 1], [0, 1], '--', color=sns.color_palette()[0])  # Add diagonal reference line
    sns.scatterplot(data=gpu_pystoi_df, x='torch_stoi_gpu', y='pystoi', hue='SNR', palette="magma", ax=axes[1])
    axes[1].set_title("torch_stoi (GPU) vs pystoi")

    plt.tight_layout()
    plt.savefig(f"stoi_comparison_duration={SAMPLE_DURATION}_batch={BATCH_SIZE}.png")
    plt.clf()


    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].set_facecolor('#f0f0f0')
    axes[0, 1].set_facecolor('#f0f0f0')
    axes[1, 0].set_facecolor('#f0f0f0')
    axes[1, 1].set_facecolor('#f0f0f0')
    # Create paired dataframes for each comparison
    cpu_pystoi_df = pd.DataFrame({
        "pystoi_STOI": [float(r) for r in pystoi_results],
        "Delta_STOI": [r1-r2 for r1, r2 in zip([float(r) for r in pystoi_results], sum([r.tolist() for r in torch_stoi_cpu_results], []))],
        'SNR': snr[:, 0].tolist()
    })

    gpu_pystoi_df = pd.DataFrame({
        "pystoi_STOI": [float(r) for r in pystoi_results],
        "Delta_STOI": [r1-r2 for r1, r2 in zip([float(r) for r in pystoi_results], sum([r.tolist() for r in torch_stoi_gpu_results], []))],
        'SNR': snr[:, 0].tolist()
    })

    # Plot using seaborn
    sns.scatterplot(data=cpu_pystoi_df, x='pystoi_STOI', y='Delta_STOI', ax=axes[0, 0])
    axes[0, 0].set_title("torch_stoi (CPU) minus pystoi")

    sns.scatterplot(data=gpu_pystoi_df, x='pystoi_STOI', y='Delta_STOI', ax=axes[0, 1])
    axes[0, 1].set_title("torch_stoi (GPU) minus pystoi")

    sns.scatterplot(data=cpu_pystoi_df, x='SNR', y='Delta_STOI', ax=axes[1, 0])
    axes[1, 0].set_title("torch_stoi (CPU) minus pystoi")

    sns.scatterplot(data=gpu_pystoi_df, x='SNR', y='Delta_STOI', ax=axes[1, 1])
    axes[1, 1].set_title("torch_stoi (GPU) minus pystoi")

    plt.tight_layout()
    plt.savefig(f"stoi_comparison_duration={SAMPLE_DURATION}_batch={BATCH_SIZE}_delta.png")
    plt.clf()


# conda install -c conda-forge pysoundfile
# pip install librosa
# import py-cpuinfo
# seaborn
# matplotlib
