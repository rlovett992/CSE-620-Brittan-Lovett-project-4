import numpy as np

from genetic_algorithm import GeneticAlgorithm


class FitnessSharing(GeneticAlgorithm):

    def __init__(
        self,
        *args,
        sharing_radius=0.05,
        sharing_alpha=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if sharing_radius <= 0:
            raise ValueError("Sharing radius must be positive.")

        if sharing_alpha <= 0:
            raise ValueError("Sharing alpha must be positive.")

        self.sharing_radius = sharing_radius
        self.sharing_alpha = sharing_alpha

    def calculate_shared_fitness(self, population):
        """Calculate fitness adjusted for population density."""

        raw_fitness = self.evaluate(population)

        # Calculate pairwise distances.
        distances = np.abs(
            population[:, np.newaxis]
            - population[np.newaxis, :]
        )

        # Calculate sharing values.
        sharing = np.where(
            distances < self.sharing_radius,
            1 - (
                distances / self.sharing_radius
            ) ** self.sharing_alpha,
            0.0,
        )

        # Calculate the niche count for each individual.
        niche_counts = np.sum(sharing, axis=1)

        # Penalize individuals in crowded niches.
        shared_fitness = raw_fitness / niche_counts

        return shared_fitness

    def next_generation(self):
        """Generate a population using fitness sharing."""

        raw_fitness = self.evaluate(self.population)

        shared_fitness = self.calculate_shared_fitness(
            self.population
        )

        # Preserve the individual with the best raw fitness.
        elite_index = np.argmax(raw_fitness)
        elite = self.population[elite_index]

        new_population = [elite]

        while len(new_population) < self.population_size:

            #Select parents using shared fitness.
            parent1 = self.selection(shared_fitness)
            parent2 = self.selection(shared_fitness)

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


if __name__ == "__main__":

    for benchmark in ["M1", "M4"]:

        ga = FitnessSharing(
            benchmark=benchmark,
            population_size=100,
            generations=200,
            crossover_rate=0.9,
            mutation_rate=0.01,
            sharing_radius=0.05,
            sharing_alpha=1.0,
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

        print("\nFinal population distribution:")

        counts, edges = np.histogram(
            result["final_population"],
            bins=10,
            range=(0, 1),
        )

        for i, count in enumerate(counts):
            print(
                f"{edges[i]:.1f} - {edges[i+1]:.1f}: "
                f"{'#' * count} ({count})"
            )