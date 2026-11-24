from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path("results")
PLOTS = Path("plots")


def test_run_count():
    runs = pd.read_csv(RESULTS / "individual_runs.csv")
    assert len(runs) == 240


def test_configuration_count():
    summary = pd.read_csv(RESULTS / "experiment_summary.csv")
    assert len(summary) == 24


def test_repetitions():
    runs = pd.read_csv(RESULTS / "individual_runs.csv")

    counts = runs.groupby(
        ["method", "benchmark", "pc", "pm"]
    ).size()

    assert (counts == 10).all()


def test_fitness_history():
    history = pd.read_csv(RESULTS / "fitness_history.csv")

    assert len(history) == 48240
    assert (history["fmax"] >= history["favg"] - 1e-10).all()
    assert (history["favg"] >= history["fmin"] - 1e-10).all()


def test_averaged_history():
    history = pd.read_csv(RESULTS / "averaged_fitness.csv")
    assert len(history) == 4824


def test_population_bounds():
    populations = pd.read_csv(RESULTS / "populations.csv")

    assert len(populations) == 48000
    assert populations["x"].between(0, 1).all()


def test_fitness_bounds():
    runs = pd.read_csv(RESULTS / "individual_runs.csv")

    assert np.isfinite(runs["best_fitness"]).all()
    assert runs["best_fitness"].between(0, 1.000001).all()


def test_peak_counts():
    runs = pd.read_csv(RESULTS / "individual_runs.csv")
    assert runs["peaks_preserved"].between(0, 5).all()


def test_best_runs():
    runs = pd.read_csv(RESULTS / "best_runs.csv")
    assert len(runs) == 6


def test_crowding_monotonicity():
    history = pd.read_csv(RESULTS / "fitness_history.csv")

    crowding = history[
        history["method"] == "Deterministic Crowding"
    ]

    for _, group in crowding.groupby("run_id"):
        group = group.sort_values("generation")

        for metric in ["fmax", "favg", "fmin"]:
            assert (np.diff(group[metric]) >= -1e-10).all()


def test_plot_count():
    assert len(list(PLOTS.glob("*.png"))) == 34