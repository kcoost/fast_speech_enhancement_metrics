import torch
from tqdm import tqdm
from datasets import load_dataset
from torchaudio.functional import resample
torch.manual_seed(42)

CACHE = False

def load_noise(num_samples: int = 1, sample_duration: int = 1, sample_rate: int = 16000):
    total_duration = num_samples * sample_duration
    flap_noise = load_dataset("nccratliri/wing-flap-noise-audio-examples", streaming=not CACHE)
    noise_list = []
    total_samples = 0
    progress_bar = tqdm(flap_noise["train"], desc="Loading noise")
    for sample in progress_bar:
        noise = resample(torch.tensor(sample["audio"]["array"]),
                        orig_freq=sample["audio"]["sampling_rate"],
                        new_freq=sample_rate)
        noise_list.append(noise)
        total_samples += len(noise)
        if total_samples >= total_duration * sample_rate:
            break
        progress_bar.set_description(f"Loading noise: {100*total_samples/(total_duration * sample_rate):.2f}%")
    
    noises = torch.cat(noise_list)
    if len(noises) < total_duration * sample_rate:
        noises = torch.cat(((total_duration * sample_rate) // len(noises) + 1)*[noises])
    noises = noises[:total_duration * sample_rate]
    noises = noises.view(num_samples, sample_duration * sample_rate)
    return noises

def load_speech(num_samples: int = 1, sample_duration: int = 1, sample_rate: int = 16000):
    total_duration = num_samples * sample_duration
    peoples_speech = load_dataset("MLCommons/peoples_speech", "validation", streaming=not CACHE)
    speech_list = []
    total_samples = 0
    progress_bar = tqdm(peoples_speech["validation"], desc="Loading speech")
    for sample in progress_bar:
        speech = resample(torch.tensor(sample["audio"]["array"]),
                      orig_freq=sample["audio"]["sampling_rate"],
                      new_freq=sample_rate)
        speech_list.append(speech)
        total_samples += len(speech)
        if total_samples >= total_duration * sample_rate:
            break
        progress_bar.set_description(f"Loading speech: {100*total_samples/(total_duration * sample_rate):.2f}%")
    
    speeches = torch.cat(speech_list)
    if len(speeches) < total_duration * sample_rate:
        speeches = torch.cat(((total_duration * sample_rate) // len(speeches) + 1)*[speeches])
    speeches = speeches[:total_duration * sample_rate]
    speeches = speeches.view(num_samples, sample_duration * sample_rate)
    return speeches

def combine_speech_noise(speech: torch.Tensor, noise: torch.Tensor, SNR_high: int = 25, SNR_low: int = -5):
    num_samples = speech.shape[0]
    speech_rms = (speech.square()).mean(dim=1, keepdim=True).sqrt()
    noise_rms = (noise.square()).mean(dim=1, keepdim=True).sqrt()
    
    snr = torch.rand(num_samples, 1) * (SNR_high - SNR_low) + SNR_low
    noise_scale = speech_rms / (10 ** (snr / 20)) / (noise_rms + 1e-12)
    noisy_speech = speech + noise_scale * noise

    return speech.float(), noisy_speech.float(), snr

def load_data(sample_duration: int = 1,
              num_samples: int = 1,
              sample_rate: int = 16000,
              SNR_high: int = 25,
              SNR_low: int = -5):
    speech = load_speech(num_samples, sample_duration, sample_rate)
    noise = load_noise(num_samples, sample_duration, sample_rate)
    speech, noisy_speech, snr = combine_speech_noise(speech, noise, SNR_high, SNR_low)
    return speech, noisy_speech, snr
