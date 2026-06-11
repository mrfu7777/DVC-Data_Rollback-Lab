"""Measure the carbon footprint of two custom MLP training setups."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from codecarbon import EmissionsTracker
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TASK_DIR = Path(__file__).resolve().parent
EMISSIONS_FILE = TASK_DIR / "emissions.csv"
METRICS_FILE = TASK_DIR / "metrics.json"
RANDOM_STATE = 42


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and split the handwritten digits dataset."""
    digits = load_digits()
    return train_test_split(
        digits.data,
        digits.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=digits.target,
    )


def build_model(hidden_layers: tuple[int, ...], max_iter: int) -> Pipeline:
    """Build a scaled MLP classifier for the selected architecture."""
    classifier = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        early_stopping=False,
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def run_experiment(
    name: str,
    hidden_layers: tuple[int, ...],
    max_iter: int,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float | int | str | list[int]]:
    """Train one model setup while CodeCarbon measures the training impact."""
    model = build_model(hidden_layers=hidden_layers, max_iter=max_iter)
    tracker = EmissionsTracker(
        project_name=name,
        output_dir=str(TASK_DIR),
        output_file=EMISSIONS_FILE.name,
        measure_power_secs=1,
        save_to_file=True,
        log_level="error",
    )

    start_time = time.perf_counter()
    tracker.start()
    model.fit(x_train, y_train)
    emissions_kg = tracker.stop()
    duration_seconds = time.perf_counter() - start_time

    probabilities = model.predict_proba(x_test)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, predictions)
    loss = log_loss(y_test, probabilities)

    return {
        "experiment": name,
        "dataset": "sklearn_digits",
        "architecture": "MLPClassifier",
        "hidden_layers": list(hidden_layers),
        "training_epochs_budget": max_iter,
        "test_accuracy": round(float(accuracy), 6),
        "test_log_loss": round(float(loss), 6),
        "co2_emitted_kg": round(float(emissions_kg or 0.0), 10),
        "duration_seconds": round(float(duration_seconds), 3),
    }


def add_carbon_roi(results: list[dict[str, float | int | str | list[int]]]) -> None:
    """Add efficiency metrics and a compact interpretation."""
    baseline = results[0]
    comparison = results[1]

    for result in results:
        emissions = float(result["co2_emitted_kg"])
        accuracy = float(result["test_accuracy"])
        result["accuracy_per_kg_co2"] = (
            round(accuracy / emissions, 2) if emissions > 0 else None
        )

    accuracy_gain = float(comparison["test_accuracy"]) - float(baseline["test_accuracy"])
    emissions_gain = float(comparison["co2_emitted_kg"]) - float(baseline["co2_emitted_kg"])

    if emissions_gain > 0 and accuracy_gain <= 0:
        interpretation = (
            "The deeper setup emitted more CO2 without improving accuracy, so the compact "
            "MLP has the better Carbon ROI for this dataset."
        )
    elif emissions_gain > 0:
        interpretation = (
            "The deeper setup improved accuracy, but the added CO2 should be justified "
            "against the size of that gain."
        )
    else:
        interpretation = (
            "The deeper setup did not emit more CO2 in this run, so both quality and "
            "emissions should be rechecked on a larger workload before deployment."
        )

    results.append(
        {
            "experiment": "comparison_summary",
            "accuracy_gain_deeper_minus_compact": round(accuracy_gain, 6),
            "co2_gain_deeper_minus_compact_kg": round(emissions_gain, 10),
            "interpretation": interpretation,
        }
    )


def main() -> None:
    """Run both Green AI experiments and save their metrics."""
    if EMISSIONS_FILE.exists():
        EMISSIONS_FILE.unlink()
    if METRICS_FILE.exists():
        METRICS_FILE.unlink()

    x_train, x_test, y_train, y_test = load_dataset()
    experiments = [
        ("compact_mlp_40_epochs", (32,), 40),
        ("deeper_mlp_120_epochs", (128, 64), 120),
    ]

    results = [
        run_experiment(
            name=name,
            hidden_layers=hidden_layers,
            max_iter=max_iter,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )
        for name, hidden_layers, max_iter in experiments
    ]
    add_carbon_roi(results)

    METRICS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for result in results[:2]:
        print(
            f"{result['experiment']} | Accuracy: {result['test_accuracy']} | "
            f"Loss: {result['test_log_loss']} | "
            f"CO2 Emitted: {result['co2_emitted_kg']} kg"
        )
    print(f"Metrics saved to: {METRICS_FILE}")
    print(f"CodeCarbon emissions saved to: {EMISSIONS_FILE}")


if __name__ == "__main__":
    main()
