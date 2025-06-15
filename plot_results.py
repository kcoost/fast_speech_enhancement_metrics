import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

SAMPLES_PER_SECOND = [
    {
        "title": "PESQ",
        "metrics": {
            "PESQ_reference_CPU_pesq": "pesq (CPU)",
            "PESQ_reference_CPU_torch_pesq": "torch_pesq (CPU)",
            "PESQ_CPU": "Optimized (CPU)",
            "PESQ_reference_GPU_torch_pesq": "torch_pesq (GPU)",
            "PESQ_GPU": "Optimized (GPU)"
        }
    },
    {
        "title": "(E)STOI",
        "metrics": {
            "STOI_reference_CPU": "pystoi (CPU)",
            "STOI_CPU": "Optimized (CPU)",
            "STOI_GPU": "Optimized (GPU)",
            
        }
    },
    {
        "title": "DNSMOS",
        "metrics": {
            "DNSMOS_reference_GPU": "espnet (GPU)",
            "DNSMOS_GPU": "Optimized (GPU)",
        }
    },
    {
        "title": "SpeechBERTScore",
        "metrics": {
            "SpeechBERTScore_reference_GPU": "discrete_speech_metrics (GPU)",
            "SpeechBERTScore_GPU": "Optimized (GPU)",
        }
    },
    {
        "title": "SDR",
        "metrics": {
            "SDR_reference_CPU": "torchmetrics (CPU)",
            "SDR_CPU": "Optimized (CPU)",
            "SDR_reference_GPU": "torchmetrics (GPU)",
            "SDR_GPU": "Optimized (GPU)",
        }
    },
    {
        "title": "LSD",
        "metrics": {
            "LSD_reference_CPU": "urgent2025 (CPU)",
            "LSD_CPU": "Optimized (CPU)",
            "LSD_GPU": "Optimized (GPU)",
        }
    },
]
DEVIATIONS = [
    {
        "title": "PESQ (CPU)",
        "metric": "PESQ",
        "metric_optimized": "PESQ_CPU",
        "metric_reference": "PESQ_reference_pesq_CPU"
    },
    {
        "title": "PESQ (GPU)",
        "metric": "PESQ",
        "metric_optimized": "PESQ_GPU",
        "metric_reference": "PESQ_reference_pesq_CPU"
    },
    {
        "title": "STOI (CPU)",
        "metric": "STOI",
        "metric_optimized": "STOI_CPU",
        "metric_reference": "STOI_reference_CPU"
    },
    {
        "title": "STOI (GPU)",
        "metric": "STOI",
        "metric_optimized": "STOI_GPU",
        "metric_reference": "STOI_reference_CPU"
    },
    {
        "title": "ESTOI (CPU)",
        "metric": "ESTOI",
        "metric_optimized": "STOI_CPU",
        "metric_reference": "STOI_reference_CPU"
    },
    {
        "title": "ESTOI (GPU)",
        "metric": "ESTOI",
        "metric_optimized": "STOI_GPU",
        "metric_reference": "STOI_reference_CPU"
    },
    {
        "title": "DNSMOS OVRL (GPU)",
        "metric": "OVRL",
        "metric_optimized": "DNSMOS_GPU",
        "metric_reference": "DNSMOS_reference_GPU"
    },
    {
        "title": "SpeechBERTScore (GPU)",
        "metric": "SpeechBERTScore",
        "metric_optimized": "SpeechBERTScore_GPU",
        "metric_reference": "SpeechBERTScore_reference_GPU"
    },
    {
        "title": "SDR (CPU)",
        "metric": "SDR",
        "metric_optimized": "SDR_CPU",
        "metric_reference": "SDR_reference_CPU"
    },
    {
        "title": "SDR (GPU)",
        "metric": "SDR",
        "metric_optimized": "SDR_GPU",
        "metric_reference": "SDR_reference_CPU"
    },
    {
        "title": "LSD (CPU)",
        "metric": "LSD",
        "metric_optimized": "LSD_CPU",
        "metric_reference": "LSD_reference_CPU"
    },
    {
        "title": "LSD (GPU)",
        "metric": "LSD",
        "metric_optimized": "LSD_GPU",
        "metric_reference": "LSD_reference_CPU"
    },
]

def load_samples_per_second():
    all_metrics = sum([list(s["metrics"].keys()) for s in SAMPLES_PER_SECOND], [])
    samples_per_second = pd.DataFrame(columns=all_metrics)
    for result_file in RESULTS_DIR.glob("*.json"):
        with open(result_file, "r") as f:
            results = json.load(f)
        for key, result in results.items():
            if key in all_metrics:
                samples_per_second[key] = results["batch_size"] / np.array(result["batch_times"])
    return samples_per_second

def load_scores():
    all_metrics = [d["metric_optimized"] for d in DEVIATIONS] + [d["metric_reference"] for d in DEVIATIONS]
    scores = pd.DataFrame()
    for result_file in RESULTS_DIR.glob("*.json"):
        with open(result_file, "r") as f:
            results = json.load(f)
        for key, result in results.items():
            if key in all_metrics:
                for metric in [d["metric"] for d in DEVIATIONS]:
                    if metric in result["values"][0]:
                        scores[f"{metric}:{key}"] = [r[metric] for r in result["values"]]
    return scores

def plot_samples_per_second(samples_per_second):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, metric_group in zip(axes, SAMPLES_PER_SECOND):
        metrics = list(metric_group["metrics"].keys())
        plot_data = pd.DataFrame()
        for metric in metrics:
            metric_data = pd.DataFrame()
            metric_data["Samples per second"] = samples_per_second[metric].values
            metric_data["Metric"] = metric_group["metrics"][metric]
            plot_data = pd.concat([plot_data, metric_data])
        plot_data = plot_data.reset_index(drop=True)
        
        sns.barplot(
            data=plot_data,
            x='Metric',
            y='Samples per second',
            ax=ax,
            errorbar='sd',
            capsize=0.1
        )
        
        ax.set_title(metric_group["title"])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel("Samples per second")
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(f"results/samples_per_second.png")

def plot_deviations(scores):    
    deviations_data = pd.DataFrame()
    for deviation in DEVIATIONS:
        metric_optimized = scores[f"{deviation['metric']}:{deviation['metric_optimized']}"]
        metric_reference = scores[f"{deviation['metric']}:{deviation['metric_reference']}"]
        deviations = (metric_optimized.values - metric_reference.values)#/metric_reference.values

        deviation_data = pd.DataFrame()
        deviation_data["Score"] = deviations#np.abs(deviations)
        deviation_data["Metric"] = deviation['title']
        #deviation_data["Title"] = deviation['title']
        deviations_data = pd.concat([deviations_data, deviation_data])

    sns.boxplot(
        deviations_data, x="Metric", y="Score",
        whis=[0, 100], width=.6, palette="vlag"
    )
    plt.xticks(rotation=90)
    # metric2 = deviation["metric"]
    # ax.plot(scores[metric1], scores[metric2], "o")
    # ax.set_title(deviation["title"])
    # ax.set_xlabel(metric1)
    # ax.set_ylabel(metric2)
    plt.tight_layout()
    plt.savefig(f"results/deviations.png")

def plot_deviations(scores):
    fig, axes = plt.subplots(len(DEVIATIONS), 1, figsize=(5, 40))

    deviations_data = pd.DataFrame()
    for deviation, ax in zip(DEVIATIONS, axes):
        metric_optimized = scores[f"{deviation['metric']}:{deviation['metric_optimized']}"]
        metric_reference = scores[f"{deviation['metric']}:{deviation['metric_reference']}"]
        deviations = (metric_optimized.values - metric_reference.values)#/metric_reference.values

        deviation_data = pd.DataFrame()
        deviation_data["Reference Score"] = metric_reference.values
        deviation_data["Optimized Score"] = metric_optimized.values
        deviation_data["Metric"] = deviation['title']
        #deviation_data["Title"] = deviation['title']
        deviations_data = pd.concat([deviations_data, deviation_data])
    
        sns.scatterplot(
            deviation_data, x="Reference Score", y="Optimized Score",
            hue="Metric",
            ax=ax
        )

    # sns.boxplot(
    #     deviations_data, x="Metric", y="Score",
    #     whis=[0, 100], width=.6, palette="vlag"
    # )
    # plt.xticks(rotation=90)
    # # metric2 = deviation["metric"]
    # # ax.plot(scores[metric1], scores[metric2], "o")
    # # ax.set_title(deviation["title"])
    # # ax.set_xlabel(metric1)
    # # ax.set_ylabel(metric2)
    plt.tight_layout()
    plt.savefig(f"results/deviations.png")

#samples_per_second = load_samples_per_second()
scores = load_scores()
#plot_samples_per_second(samples_per_second)
plot_deviations(scores)




# samples_per_second = pd.DataFrame()
# optimized_metrics = []
# reference_metrics = []
# for key, result in results.items():
#     if isinstance(result, dict):
#         if "reference" in key:
#             reference_metrics.append(key)
#         else:
#             optimized_metrics.append(key)
#         samples_per_second[key] = results["batch_size"] / np.array(result["batch_times"])

# deviations = pd.DataFrame()
# for metric in optimized_metrics:
#     deviations[metric] = samples_per_second[metric] / samples_per_second[metric.replace("optimized", "reference")]

# print(2)

# sns.lineplot(x="SNR_high", y="SpeechBERTScore", data=samples_per_second)
# plt.show()