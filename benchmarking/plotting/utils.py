import json

import numpy as np
import pandas as pd
from benchmarking.plotting.settings import SAMPLES_PER_SECOND


def load_results(results_dir):
    all_metrics = sum([s["metrics"] for s in SAMPLES_PER_SECOND], [])
    results = []
    for result_file in results_dir.rglob("*.json"):
        with open(result_file, "r") as f:
            result_data = json.load(f)
        for key, data in result_data.items():
            if key in all_metrics:
                result = pd.DataFrame(columns=["Samples per second", "Metric"])
                result["Samples per second"] = result_data["batch_size"] / np.array(data["batch_times"])
                result["Metric"] = key
                result["Batch size"] = result_data["batch_size"]
                results.append(result)
    return pd.concat(results)


def load_values(results_dir, settings):
    name = settings["name"]
    with open(results_dir / f"{name}_results.json", "r") as f:
        results = json.load(f)

    reference_data = results[settings["metric_reference"]]
    optimized_data = results[settings["metric_optimized"]]

    reference_values = [v[settings["metric"]] for v in reference_data["values"]]
    optimized_values = [v[settings["metric"]] for v in optimized_data["values"]]

    return optimized_values, reference_values
