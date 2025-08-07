import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from benchmarking.plotting.settings import NAME_MAPPING, PLOTS_DIR, RESULTS_DIR, SAMPLES_PER_SECOND
from benchmarking.plotting.utils import load_results


def plot_samples_per_second(results_dir):
    results = load_results(results_dir)
    _, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, metric_group in zip(axes.flatten(), SAMPLES_PER_SECOND, strict=False):
        metrics = metric_group["metrics"]
        plot_data = results[results["Metric"].isin(metrics)]

        color_mapping = {k: v[2] for k, v in NAME_MAPPING.items()}
        metric_labels = [NAME_MAPPING[m][0] for m in metrics]
        device_labels = [NAME_MAPPING[m][1] for m in metrics]

        sns.barplot(
            data=plot_data,
            x="Metric",
            y="Samples per second",
            ax=ax,
            order=metrics,
            palette=color_mapping,
        )

        ax.set_xticklabels(metric_labels)

        for tick, device in zip(ax.get_xticks(), device_labels, strict=False):
            bar = [b for b in ax.patches if abs((b.get_x() + b.get_width() / 2) - tick) < 1e-6]
            if bar:
                height = bar[0].get_height()
                label_y = height * 1.05
                ax.text(
                    tick,
                    label_y,
                    device,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    rotation=0,
                    clip_on=False,
                )

        ax.set_title(metric_group["title"], fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels())
        ax.set_ylabel("Samples/sec", fontsize=12)
        ax.set_yscale("log")

        y_min = np.floor(np.log10(0.8 * plot_data["Samples per second"].min()))
        y_max = np.ceil(np.log10(plot_data["Samples per second"].max()))
        ax.set_ylim(10 ** int(y_min), 10 ** int(y_max))
        yticks = [10**i for i in range(int(y_min), int(y_max) + 1)]
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(int(y)) for y in yticks])
        ax.set_xlabel("")
        ax.tick_params(axis="x", length=0, labelsize=10)

        ax.set_facecolor("#f7f7f7")

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "samples_per_second.png")


if __name__ == "__main__":
    plot_samples_per_second(RESULTS_DIR)
