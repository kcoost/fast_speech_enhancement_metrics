from pathlib import Path
import seaborn as sns

PLOTS_DIR = Path(__file__).parents[2] / "plots"
ALL_RESULTS_DIR = Path(__file__).parents[2] / "results"
RESULTS_DIR = ALL_RESULTS_DIR / "batch_size_64"

SAMPLES_PER_SECOND = [
    {
        "title": "PESQ",
        "metrics": [
            "PESQ_reference_pesq_CPU",
            "PESQ_reference_torch_pesq_CPU",
            "PESQ_reference_torch_pesq_GPU",
            "PESQ_CPU",
            "PESQ_GPU",
        ],
    },
    {
        "title": "(E)STOI",
        "metrics": [
            "STOI_reference_CPU",
            "STOI_CPU",
            "STOI_GPU",
        ],
    },
    {
        "title": "DNSMOS",
        "metrics": [
            "DNSMOS_reference_GPU",
            "DNSMOS_GPU",
        ],
    },
    {
        "title": "SpeechBERTScore",
        "metrics": [
            "SpeechBERTScore_reference_GPU",
            "SpeechBERTScore_GPU",
        ],
    },
    {
        "title": "SDR",
        "metrics": [
            "SDR_reference_CPU",
            "SDR_reference_GPU",
            "SDR_CPU",
            "SDR_GPU",
        ],
    },
    {
        "title": "LSD",
        "metrics": [
            "LSD_reference_CPU",
            "LSD_CPU",
            "LSD_GPU",
        ],
    },
]

# Maps name to (title, device, color)
CPU_PALETTE = sns.color_palette("pastel")
GPU_PALETTE = sns.color_palette("tab10")
NAME_MAPPING = {
    "PESQ_reference_pesq_CPU": ("pesq", "CPU", CPU_PALETTE[9]),
    "PESQ_reference_torch_pesq_CPU": ("torch_pesq", "CPU", CPU_PALETTE[0]),
    "PESQ_CPU": ("Ours", "CPU", CPU_PALETTE[4]),
    "PESQ_reference_torch_pesq_GPU": ("torch_pesq", "GPU", GPU_PALETTE[0]),
    "PESQ_GPU": ("Ours", "GPU", GPU_PALETTE[4]),
    "STOI_reference_CPU": ("pystoi", "CPU", CPU_PALETTE[1]),
    "STOI_CPU": ("Ours", "CPU", CPU_PALETTE[4]),
    "STOI_GPU": ("Ours", "GPU", GPU_PALETTE[4]),
    "DNSMOS_reference_GPU": ("espnet", "GPU", GPU_PALETTE[5]),
    "DNSMOS_GPU": ("Ours", "GPU", GPU_PALETTE[4]),
    "SpeechBERTScore_reference_GPU": ("discrete_speech_metrics", "GPU", GPU_PALETTE[3]),
    "SpeechBERTScore_GPU": ("Ours", "GPU", GPU_PALETTE[4]),
    "SDR_reference_CPU": ("torchmetrics", "CPU", CPU_PALETTE[2]),
    "SDR_CPU": ("Ours", "CPU", CPU_PALETTE[4]),
    "SDR_reference_GPU": ("torchmetrics", "GPU", GPU_PALETTE[2]),
    "SDR_GPU": ("Ours", "GPU", GPU_PALETTE[4]),
    "LSD_reference_CPU": ("urgent2025", "CPU", CPU_PALETTE[6]),
    "LSD_CPU": ("Ours", "CPU", CPU_PALETTE[4]),
    "LSD_GPU": ("Ours", "GPU", GPU_PALETTE[4]),
}

DEVIATION_SETTINGS = [
    {
        "name": "PESQ",
        "metric": "PESQ",
        "metric_optimized": "PESQ_GPU",
        "metric_reference": "PESQ_reference_torch_pesq_GPU",  # "PESQ_reference_pesq_CPU",#
    },
    {
        "name": "STOI",
        "metric": "STOI",
        "metric_optimized": "STOI_GPU",
        "metric_reference": "STOI_reference_CPU",
    },
    # {
    #     "title": "ESTOI (GPU)",
    #     "metric": "ESTOI",
    #     "metric_optimized": "STOI_GPU",
    #     "metric_reference": "STOI_reference_CPU",
    # },
    {
        "name": "DNSMOS",
        "metric": "OVRL",
        "metric_optimized": "DNSMOS_GPU",
        "metric_reference": "DNSMOS_reference_GPU",
    },
    {
        "name": "SpeechBERTScore",
        "metric": "SpeechBERTScore",
        "metric_optimized": "SpeechBERTScore_GPU",
        "metric_reference": "SpeechBERTScore_reference_GPU",
    },
    {
        "name": "SDR",
        "metric": "SDR",
        "metric_optimized": "SDR_GPU",
        "metric_reference": "SDR_reference_CPU",
    },
    {
        "name": "LSD",
        "metric": "LSD",
        "metric_optimized": "LSD_GPU",
        "metric_reference": "LSD_reference_CPU",
    },
]
