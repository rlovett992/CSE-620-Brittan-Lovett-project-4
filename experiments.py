import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks import get_peaks, evaluate_population
from genetic_algorithm import GeneticAlgorithm
from fitness_sharing import FitnessSharing
from deterministic_crowding import DeterministicCrowding


RESULTS_DIR = Path("results")

METHODS = {
    "Standard GA": GeneticAlgorithm,
    "Fitness Sharing": FitnessSharing,
    "Deterministic Crowding": DeterministicCrowding,
}

BENCHMARKS = ["M1", "M4"]

CROSSOVER_RATES = [0.60, 0.90]
MUTATION_RATES = [0.01, 0.10]

POPULATION_SIZE = 100
GENERATIONS = 200
REPETITIONS = 10

BASE_SEED = 42

PEAK_DISTANCE_TOLERANCE = 0.02
PEAK_FITNESS_THRESHOLD = 0.90


def count_preserved_peaks(population, benchmark):
    """Count peaks represented by high-fitness individuals."""

    locations, heights = get_peaks(benchmark)

    fitness = evaluate_population(
        population,
        benchmark,
    )

    preserved = 0

    for location, height in zip(locations, heights):

        nearby = (
            np.abs(population - location)
            <= PEAK_DISTANCE_TOLERANCE
        )

        sufficiently_fit = (
            fitness >= height * PEAK_FITNESS_THRESHOLD
        )

        if np.any(nearby & sufficiently_fit):
            preserved += 1

    return preserved


def run_experiments():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_records = []
    history_records = []
    population_records = []

    configurations = list(itertools.product(
        METHODS.keys(),
        BENCHMARKS,
        CROSSOVER_RATES,
        MUTATION_RATES,
    ))

    total_runs = len(configurations) * REPETITIONS
    completed = 0

    start_time = time.perf_counter()

    for method, benchmark, pc, pm in configurations:

        algorithm_class = METHODS[method]

        for repetition in range(REPETITIONS):

            seed = BASE_SEED + repetition

            algorithm = algorithm_class(
                benchmark=benchmark,
                population_size=POPULATION_SIZE,
                generations=GENERATIONS,
                crossover_rate=pc,
                mutation_rate=pm,
                seed=seed,
            )

            run_start = time.perf_counter()

            result = algorithm.run()

            runtime = time.perf_counter() - run_start

            run_id = completed + 1

            metadata = {
                "run_id": run_id,
                "method": method,
                "benchmark": benchmark,
                "pc": pc,
                "pm": pm,
                "repetition": repetition + 1,
                "seed": seed,
            }

            peaks = count_preserved_peaks(
                result["final_population"],
                benchmark,
            )

            history = result["history"]

            run_records.append({
                **metadata,
                "best_solution": result["best_solution"],
                "best_fitness": result["best_fitness"],
                "final_fmax": history["fmax"][-1],
                "final_favg": history["favg"][-1],
                "final_fmin": history["fmin"][-1],
                "peaks_preserved": peaks,
                "runtime_seconds": runtime,
            })

            for i, generation in enumerate(
                history["generation"]
            ):

                history_records.append({
                    **metadata,
                    "generation": generation,
                    "fmax": history["fmax"][i],
                    "favg": history["favg"][i],
                    "fmin": history["fmin"][i],
                })

            for stage in ["initial", "final"]:

                population = result[
                    f"{stage}_population"
                ]

                fitness = evaluate_population(
                    population,
                    benchmark,
                )

                for individual, value in zip(
                    population,
                    fitness,
                ):

                    population_records.append({
                        **metadata,
                        "stage": stage,
                        "x": individual,
                        "fitness": value,
                    })

            completed += 1

            print(
                f"[{completed}/{total_runs}] "
                f"{method} | {benchmark} | "
                f"Pc={pc:.2f} | Pm={pm:.2f} | "
                f"Run={repetition + 1} | "
                f"Fitness={result['best_fitness']:.6f} | "
                f"Peaks={peaks} | "
                f"Time={runtime:.2f}s"
            )

    runs = pd.DataFrame(run_records)
    histories = pd.DataFrame(history_records)
    populations = pd.DataFrame(population_records)

    group_columns = [
        "method",
        "benchmark",
        "pc",
        "pm",
    ]

    averaged_histories = histories.groupby(
        group_columns + ["generation"],
        as_index=False,
    )[["fmax", "favg", "fmin"]].mean()

    summary = runs.groupby(
        group_columns,
        as_index=False,
    ).agg(
        mean_best_fitness=("best_fitness", "mean"),
        std_best_fitness=("best_fitness", "std"),
        max_best_fitness=("best_fitness", "max"),
        mean_final_fmax=("final_fmax", "mean"),
        mean_final_favg=("final_favg", "mean"),
        mean_final_fmin=("final_fmin", "mean"),
        mean_peaks_preserved=("peaks_preserved", "mean"),
        max_peaks_preserved=("peaks_preserved", "max"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
    )

    # Select the best run for each method and benchmark.
    # Prioritize peak preservation, then best fitness.
    best_runs = runs.sort_values(
        ["peaks_preserved", "best_fitness"],
        ascending=[False, False],
    ).groupby(
        ["method", "benchmark"],
        as_index=False,
    ).head(1)

    runs.to_csv(
        RESULTS_DIR / "individual_runs.csv",
        index=False,
    )

    histories.to_csv(
        RESULTS_DIR / "fitness_history.csv",
        index=False,
    )

    populations.to_csv(
        RESULTS_DIR / "populations.csv",
        index=False,
    )

    averaged_histories.to_csv(
        RESULTS_DIR / "averaged_fitness.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS_DIR / "experiment_summary.csv",
        index=False,
    )

    best_runs.to_csv(
        RESULTS_DIR / "best_runs.csv",
        index=False,
    )

    elapsed = time.perf_counter() - start_time

    print("\nAll experiments completed.")
    print(f"Total runs: {completed}")
    print(f"Total runtime: {elapsed:.2f} seconds")
    print(f"Results saved to: {RESULTS_DIR.resolve()}")

    return summary


if __name__ == "__main__":
    run_experiments()