import numpy as np

LOWER_BOUND = 0.0
UPPER_BOUND = 1.0


def m1(x):
    """Multimodal function with five equal-height peaks."""
    x = np.asarray(x, dtype=float)

    return np.sin(5 * np.pi * x) ** 6


def m4(x):
    """Multimodal function with unevenly spaced, unequal peaks."""
    x = np.asarray(x, dtype=float)

    envelope = np.exp(
        -2 * np.log(2) * ((x - 0.08) / 0.854) ** 2
    )

    oscillation = np.sin(
        5 * np.pi * (x ** 0.75 - 0.05)
    ) ** 6

    return envelope * oscillation


BENCHMARKS = {
    "M1": m1,
    "M4": m4,
}


def evaluate_population(population, benchmark):
    """Evaluate the fitness of every individual."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    return BENCHMARKS[benchmark](population)


def get_peaks(benchmark, resolution=100001):
    """Numerically identify the benchmark's local maxima."""
    x = np.linspace(LOWER_BOUND, UPPER_BOUND, resolution)
    y = BENCHMARKS[benchmark](x)

    indices = np.where(
        (y[1:-1] > y[:-2]) &
        (y[1:-1] > y[2:])
    )[0] + 1

    return x[indices], y[indices]


if __name__ == "__main__":
    for name in BENCHMARKS:
        locations, fitness = get_peaks(name)

        print(f"\nBenchmark: {name}")
        print(f"Number of peaks: {len(locations)}")

        for i, (x, f) in enumerate(
            zip(locations, fitness), start=1
        ):
            print(f"Peak {i}: x={x:.6f}, fitness={f:.6f}")