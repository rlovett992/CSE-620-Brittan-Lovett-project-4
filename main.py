import argparse
import time

from experiments import run_experiments
from plotting import generate_all_plots


def main():

    parser = argparse.ArgumentParser(
        description="Genetic Algorithm Niching Comparison"
    )

    parser.add_argument(
        "--experiments-only",
        action="store_true",
        help="Run experiments without generating plots.",
    )

    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Generate plots using existing results.",
    )

    args = parser.parse_args()

    if args.experiments_only and args.plots_only:
        parser.error(
            "These options cannot be used together."
        )

    start_time = time.perf_counter()

    print("=" * 60)
    print("GENETIC ALGORITHM NICHING COMPARISON")
    print("=" * 60)

    if not args.plots_only:

        print("\nStarting experiments...\n")

        summary = run_experiments()

        print("\nExperiment Summary:")
        print(summary.to_string(index=False))

    if not args.experiments_only:

        print("\nStarting plot generation...\n")

        generate_all_plots()

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print(f"Total runtime: {elapsed:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()