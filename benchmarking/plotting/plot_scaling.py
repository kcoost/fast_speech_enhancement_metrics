import matplotlib.pyplot as plt
import seaborn as sns

from benchmarking.plotting.settings import NAME_MAPPING, PLOTS_DIR, SAMPLES_PER_SECOND, ALL_RESULTS_DIR
from benchmarking.plotting.utils import load_results


def plot_samples_per_second_scaling(results_dir):
    results = load_results(results_dir)
    _, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, metric_group in zip(axes.flatten(), SAMPLES_PER_SECOND, strict=False):
        data_subset = results[results["Metric"].isin(metric_group["metrics"])]
        plot_data = data_subset.groupby(["Batch size", "Metric"]).mean().reset_index()

        color_mapping = {k: v[2] for k, v in NAME_MAPPING.items()}
        sns.lineplot(data=plot_data, x="Batch size", y="Samples per second", hue="Metric", ax=ax, palette=color_mapping)

        ax.set_title(metric_group["title"], fontsize=14)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel("Samples per second", fontsize=12)
        ax.set_xlabel("Batch size", fontsize=12)
        ax.set_yscale("log")
        ax.set_facecolor("#f7f7f7")

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "samples_per_second_scaling.png")


if __name__ == "__main__":
    plot_samples_per_second_scaling(ALL_RESULTS_DIR)
