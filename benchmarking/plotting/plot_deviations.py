import seaborn as sns
import matplotlib.pyplot as plt

from benchmarking.plotting.settings import DEVIATION_SETTINGS, PLOTS_DIR, RESULTS_DIR, NAME_MAPPING
from benchmarking.plotting.utils import load_values


def plot_deviations(results_dir, deviation_settings):
    _, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, settings in zip(axes.flatten(), deviation_settings, strict=False):
        optimized_values, reference_values = load_values(results_dir, settings)

        name_optimized, device_optimized, _ = NAME_MAPPING[settings["metric_optimized"]]
        name_reference, device_reference, color = NAME_MAPPING[settings["metric_reference"]]

        sns.scatterplot(x=optimized_values, y=reference_values, ax=ax, color=color)
        ax.set_title(settings["name"], fontsize=14)

        ax.set_xlabel(f"{name_optimized} ({device_optimized})", fontsize=12)
        ax.set_ylabel(f"{name_reference} ({device_reference})", fontsize=12)

        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.tick_params(axis="both", which="minor", labelsize=12)
        ax.set_facecolor("#f7f7f7")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "deviations.png")


if __name__ == "__main__":
    plot_deviations(RESULTS_DIR, DEVIATION_SETTINGS)
