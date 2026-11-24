# Genetic Algorithm Niching Comparison

## Overview

This project implements and compares three evolutionary optimization algorithms on two multimodal benchmark functions, M1 and M4.

The objective is to evaluate the effectiveness of niching techniques in maintaining population diversity, preserving multiple local optima, and improving optimization performance.

## Implemented Algorithms

- **Standard Genetic Algorithm:** Roulette-wheel selection, arithmetic crossover, Gaussian mutation, and elitism.
- **Fitness Sharing:** Adjusts individual fitness according to population density to encourage diversity.
- **Deterministic Crowding:** Preserves diversity through competition between genetically similar parents and offspring.

## Benchmark Functions

- **M1:** Five equally spaced peaks of equal height.
- **M4:** Five unevenly spaced peaks of unequal height.

Both functions use a search domain of [0, 1] and are treated as maximization problems.

The benchmark definitions were selected from conventional multimodal optimization formulations and have not been verified against the original course lecture slides.

## Experimental Configuration

| Parameter | Value |
|---|---|
| Population Size | 100 |
| Generations | 200 |
| Crossover Probability | 0.60, 0.90 |
| Mutation Probability | 0.01, 0.10 |
| Mutation Standard Deviation | 0.05 |
| Sharing Radius | 0.05 |
| Sharing Exponent | 1.0 |
| Repetitions per Configuration | 10 |
| Random Seeds | 42–51 |
| Total Configurations | 24 |
| Total Experimental Runs | 240 |

## Project Structure

| File | Description |
|---|---|
| benchmarks.py | Benchmark functions and peak detection |
| genetic_algorithm.py | Standard genetic algorithm |
| fitness_sharing.py | Fitness sharing implementation |
| deterministic_crowding.py | Deterministic crowding implementation |
| experiments.py | Automated experimental execution |
| plotting.py | Experimental visualization |
| main.py | Main program entry point |
| requirements.txt | Python dependencies |
| tests/ | Automated result validation |
| results/ | Experimental CSV data |
| plots/ | Generated figures |

## Installation

Create a Python virtual environment:

    python -m venv venv

Activate the environment on Windows:

    .\venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install -r requirements.txt

## Execution

Run all experiments and generate plots:

    python main.py

Run only the experiments:

    python main.py --experiments-only

Regenerate plots from existing experimental data:

    python main.py --plots-only

Validate experimental results:

    python -m pytest tests -v

## Experimental Results

The project is designed to generate six CSV files:

- individual_runs.csv
- fitness_history.csv
- populations.csv
- averaged_fitness.csv
- experiment_summary.csv
- best_runs.csv

The visualization system generates 34 figures:

- 24 fitness convergence plots.
- 6 initial and final population landscape plots.
- 2 algorithm comparison plots.
- 2 peak preservation plots.

## Evaluation Metrics

The algorithms are evaluated using:

- Maximum population fitness (Fmax).
- Average population fitness (Favg).
- Minimum population fitness (Fmin).
- Number of preserved peaks.
- Best solution fitness.
- Execution time.

A peak is considered preserved when at least one individual is within 0.02 of its location and achieves at least 90% of its fitness.

Best runs are selected primarily by peak preservation, with best fitness used as a tiebreaker.

All fitness statistics are calculated using raw benchmark fitness to ensure consistent comparisons between algorithms.

## Reproducibility

Each experimental configuration uses the same ten random seeds. Population size, generation count, crossover operators, and mutation operators are consistent across all three algorithms.