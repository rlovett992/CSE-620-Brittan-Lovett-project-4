import numpy as np

from benchmarks import (
    LOWER_BOUND,
    UPPER_BOUND,
    evaluate_population,
)


class GeneticAlgorithm:

    def __init__(
        self,
        benchmark,
        population_size=100,
        generations=200,
        crossover_rate=0.9,
        mutation_rate=0.01,
        mutation_sigma=0.05,
        seed=None,
    ):
        self.benchmark = benchmark
        self.population_size = population_size
        self.generations = generations

        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma

        self.rng = np.random.default_rng(seed)

        self.population = None
        self.initial_population = None

        self.history = {
            "generation": [],
            "fmax": [],
            "favg": [],
            "fmin": [],
        }

    def initialize_population(self):
        """Generate a uniformly distributed population."""

        self.population = self.rng.uniform(
            LOWER_BOUND,
            UPPER_BOUND,
            self.population_size,
        )

        self.initial_population = self.population.copy()

    def evaluate(self, population):
        """Evaluate raw benchmark fitness."""

        return evaluate_population(
            population,
            self.benchmark,
        )

    def selection(self, fitness):
        """Roulette-wheel parent selection."""

        total_fitness = np.sum(fitness)

        if total_fitness <= 0:
            return self.rng.choice(self.population)

        probabilities = fitness / total_fitness

        index = self.rng.choice(
            len(self.population),
            p=probabilities,
        )

        return self.population[index]

    def crossover(self, parent1, parent2):
        """Arithmetic crossover."""

        if self.rng.random() >= self.crossover_rate:
            return parent1, parent2

        alpha = self.rng.random()

        child1 = (
            alpha * parent1
            + (1 - alpha) * parent2
        )

        child2 = (
            alpha * parent2
            + (1 - alpha) * parent1
        )

        return child1, child2

    def mutate(self, individual):
        """Gaussian mutation."""

        if self.rng.random() < self.mutation_rate:

            individual += self.rng.normal(
                0,
                self.mutation_sigma,
            )

        return float(
            np.clip(
                individual,
                LOWER_BOUND,
                UPPER_BOUND,
            )
        )

    def record_statistics(self, generation):
        """Record raw fitness statistics."""

        fitness = self.evaluate(self.population)

        self.history["generation"].append(generation)
        self.history["fmax"].append(float(np.max(fitness)))
        self.history["favg"].append(float(np.mean(fitness)))
        self.history["fmin"].append(float(np.min(fitness)))

    def next_generation(self):
        """Produce the next population."""

        fitness = self.evaluate(self.population)

        # Preserve the best individual.
        elite_index = np.argmax(fitness)
        elite = self.population[elite_index]

        new_population = [elite]

        while len(new_population) < self.population_size:

            parent1 = self.selection(fitness)
            parent2 = self.selection(fitness)

            child1, child2 = self.crossover(
                parent1,
                parent2,
            )

            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            new_population.append(child1)

            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = np.asarray(
            new_population,
            dtype=float,
        )

    def run(self):
        """Execute the genetic algorithm."""

        for values in self.history.values():
            values.clear()

        self.initialize_population()
        self.record_statistics(0)

        for generation in range(1, self.generations + 1):

            self.next_generation()
            self.record_statistics(generation)

        fitness = self.evaluate(self.population)
        best_index = np.argmax(fitness)

        return {
            "benchmark": self.benchmark,
            "method": self.__class__.__name__,
            "seed": self.seed_value,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "best_solution": float(
                self.population[best_index]
            ),
            "best_fitness": float(
                fitness[best_index]
            ),
            "initial_population":
                self.initial_population.copy(),
            "final_population":
                self.population.copy(),
            "history": {
                key: values.copy()
                for key, values in self.history.items()
            },
        }

    @property
    def seed_value(self):
        """Return the original RNG seed when available."""
        return self.rng.bit_generator.seed_seq.entropy


if __name__ == "__main__":

    for benchmark in ["M1", "M4"]:

        ga = GeneticAlgorithm(
            benchmark=benchmark,
            population_size=100,
            generations=200,
            crossover_rate=0.9,
            mutation_rate=0.01,
            seed=42,
        )

        result = ga.run()

        print(f"\nBenchmark: {benchmark}")
        print(f"Method: {result['method']}")
        print(f"Best solution: {result['best_solution']:.6f}")
        print(f"Best fitness: {result['best_fitness']:.6f}")
        print(f"Final Fmax: {result['history']['fmax'][-1]:.6f}")
        print(f"Final Favg: {result['history']['favg'][-1]:.6f}")
        print(f"Final Fmin: {result['history']['fmin'][-1]:.6f}")