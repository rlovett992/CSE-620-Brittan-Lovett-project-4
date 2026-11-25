from pathlib import Path

import numpy as np
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from benchmarks import BENCHMARKS, get_peaks


RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

METHOD_COLORS = {
    "Standard GA": "tab:blue",
    "Fitness Sharing": "tab:orange",
    "Deterministic Crowding": "tab:green",
}


def safe_name(name):
    return name.lower().replace(" ", "_")


def save_plot(fig, filename):
    fig.tight_layout()

    fig.savefig(
        PLOTS_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_fitness_convergence():

    data = pd.read_csv(
        RESULTS_DIR / "averaged_fitness.csv"
    )

    groups = data.groupby(
        ["method", "benchmark", "pc", "pm"]
    )

    for (method, benchmark, pc, pm), group in groups:

        group = group.sort_values("generation")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            group["generation"],
            group["fmax"],
            label="Fmax",
            color="tab:green",
        )

        ax.plot(
            group["generation"],
            group["favg"],
            label="Favg",
            color="tab:blue",
        )

        ax.plot(
            group["generation"],
            group["fmin"],
            label="Fmin",
            color="tab:red",
        )

        ax.set_title(
            f"{method} - {benchmark}\n"
            f"Pc={pc:.2f}, Pm={pm:.2f} "
            f"(Average of 10 Runs)"
        )

        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)

        filename = (
            f"convergence_{safe_name(method)}_"
            f"{benchmark}_pc{pc:.2f}_pm{pm:.2f}.png"
        )

        save_plot(fig, filename)


def plot_population_landscapes():

    populations = pd.read_csv(
        RESULTS_DIR / "populations.csv"
    )

    best_runs = pd.read_csv(
        RESULTS_DIR / "best_runs.csv"
    )

    x = np.linspace(0, 1, 5000)

    for _, run in best_runs.iterrows():

        method = run["method"]
        benchmark = run["benchmark"]
        run_id = run["run_id"]

        y = BENCHMARKS[benchmark](x)

        peak_locations, peak_heights = get_peaks(
            benchmark
        )

        population = populations[
            populations["run_id"] == run_id
        ]

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 9),
            sharex=True,
            sharey=True,
        )

        for ax, stage in zip(
            axes,
            ["initial", "final"],
        ):

            stage_data = population[
                population["stage"] == stage
            ]

            ax.plot(
                x,
                y,
                color="black",
                linewidth=1.5,
                label="Benchmark Function",
            )

            ax.scatter(
                stage_data["x"],
                stage_data["fitness"],
                color=(
                    "tab:blue"
                    if stage == "initial"
                    else "tab:red"
                ),
                s=25,
                alpha=0.65,
                label=f"{stage.capitalize()} Population",
                zorder=3,
            )

            ax.scatter(
                peak_locations,
                peak_heights,
                marker="*",
                color="gold",
                edgecolors="black",
                s=120,
                label="Known Peaks",
                zorder=4,
            )

            ax.set_title(
                f"{stage.capitalize()} Population"
            )

            ax.set_ylabel("Fitness")
            ax.set_ylim(0, 1.08)
            ax.legend(loc="lower left")
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Solution (x)")

        fig.suptitle(
            f"{method} - {benchmark}\n"
            f"Pc={run['pc']:.2f}, "
            f"Pm={run['pm']:.2f}, "
            f"Run={int(run['repetition'])}\n"
            f"Peaks Preserved: "
            f"{int(run['peaks_preserved'])}"
        )

        filename = (
            f"population_{safe_name(method)}_"
            f"{benchmark}.png"
        )

        save_plot(fig, filename)


def get_best_configurations(summary):

    return summary.sort_values(
        [
            "mean_peaks_preserved",
            "mean_best_fitness",
        ],
        ascending=[False, False],
    ).groupby(
        ["method", "benchmark"],
        as_index=False,
    ).head(1)


def plot_algorithm_comparison():

    history = pd.read_csv(
        RESULTS_DIR / "averaged_fitness.csv"
    )

    summary = pd.read_csv(
        RESULTS_DIR / "experiment_summary.csv"
    )

    best_configs = get_best_configurations(summary)

    for benchmark in BENCHMARKS:

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(11, 12),
            sharex=True,
        )

        configurations = best_configs[
            best_configs["benchmark"] == benchmark
        ]

        for _, config in configurations.iterrows():

            method = config["method"]

            data = history[
                (history["benchmark"] == benchmark)
                & (history["method"] == method)
                & np.isclose(
                    history["pc"], config["pc"]
                )
                & np.isclose(
                    history["pm"], config["pm"]
                )
            ].sort_values("generation")

            label = (
                f"{method} "
                f"(Pc={config['pc']:.2f}, "
                f"Pm={config['pm']:.2f})"
            )

            for ax, metric in zip(
                axes,
                ["fmax", "favg", "fmin"],
            ):

                ax.plot(
                    data["generation"],
                    data[metric],
                    label=label,
                    color=METHOD_COLORS[method],
                )

        for ax, metric in zip(
            axes,
            ["Fmax", "Favg", "Fmin"],
        ):

            ax.set_title(metric)
            ax.set_ylabel("Fitness")
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Generation")

        fig.suptitle(
            f"Algorithm Comparison - {benchmark}\n"
            "Best Configuration of Each Method"
        )

        save_plot(
            fig,
            f"algorithm_comparison_{benchmark}.png",
        )


def plot_peak_preservation():

    summary = pd.read_csv(
        RESULTS_DIR / "experiment_summary.csv"
    )

    for benchmark in BENCHMARKS:

        data = summary[
            summary["benchmark"] == benchmark
        ]

        configurations = sorted(
            set(zip(data["pc"], data["pm"]))
        )

        methods = list(METHOD_COLORS.keys())

        positions = np.arange(len(configurations))
        width = 0.25

        fig, ax = plt.subplots(figsize=(11, 6))

        for i, method in enumerate(methods):

            values = []

            for pc, pm in configurations:

                row = data[
                    (data["method"] == method)
                    & np.isclose(data["pc"], pc)
                    & np.isclose(data["pm"], pm)
                ]

                values.append(
                    row["mean_peaks_preserved"].iloc[0]
                )

            bars = ax.bar(
                positions + (i - 1) * width,
                values,
                width,
                label=method,
                color=METHOD_COLORS[method],
            )

            ax.bar_label(
                bars,
                fmt="%.2f",
                fontsize=9,
                padding=3,
            )

        peak_locations, _ = get_peaks(benchmark)

        ax.set_title(
            f"Average Peak Preservation - {benchmark}"
        )

        ax.set_ylabel("Average Peaks Preserved")

        ax.set_xticks(
            positions,
            [
                f"Pc={pc:.2f}\nPm={pm:.2f}"
                for pc, pm in configurations
            ],
        )

        ax.set_ylim(0, len(peak_locations) + 0.5)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        save_plot(
            fig,
            f"peak_preservation_{benchmark}.png",
        )


def generate_all_plots():

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Generating fitness convergence plots...")
    plot_fitness_convergence()

    print("Generating population landscape plots...")
    plot_population_landscapes()

    print("Generating algorithm comparison plots...")
    plot_algorithm_comparison()

    print("Generating peak preservation plots...")
    plot_peak_preservation()

    print("\nAll plots generated.")
    print(f"Plots saved to: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    generate_all_plots()