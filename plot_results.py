from pathlib import Path
from benchmarking.plotting.plot_deviations import plot_deviations
from benchmarking.plotting.plot_samples_per_second import plot_samples_per_second
from benchmarking.plotting.plot_scaling import plot_samples_per_second_scaling
from benchmarking.plotting.settings import DEVIATION_SETTINGS, RESULTS_DIR, ALL_RESULTS_DIR


if __name__ == "__main__":
    assert (Path(__file__).parent / "results").exists(), "No results found, run benchmark_metrics.py first"

    plot_samples_per_second(RESULTS_DIR)
    plot_deviations(RESULTS_DIR, DEVIATION_SETTINGS)
    plot_samples_per_second_scaling(ALL_RESULTS_DIR)
