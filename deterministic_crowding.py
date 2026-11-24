import numpy as np

from genetic_algorithm import GeneticAlgorithm


class DeterministicCrowding(GeneticAlgorithm):

    def compete(self, parent, child):
        """Return the fitter individual."""

        parent_fitness = self.evaluate(parent)
        child_fitness = self.evaluate(child)

        if child_fitness >= parent_fitness:
            return child

        return parent

    def next_generation(self):
        """Generate a population using deterministic crowding."""

        # Randomize the order of the population.
        indices = self.rng.permutation(
            self.population_size
        )

        shuffled_population = self.population[indices]

        new_population = []

        # Process individuals in pairs.
        for i in range(0, self.population_size - 1, 2):

            parent1 = shuffled_population[i]
            parent2 = shuffled_population[i + 1]

            # Generate offspring.
            child1, child2 = self.crossover(
                parent1,
                parent2,
            )

            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            # Calculate direct pairing distance.
            distance1 = (
                abs(parent1 - child1)
                + abs(parent2 - child2)
            )

            # Calculate crossed pairing distance.
            distance2 = (
                abs(parent1 - child2)
                + abs(parent2 - child1)
            )

            # Select the most similar pairing.
            if distance1 <= distance2:

                survivor1 = self.compete(
                    parent1,
                    child1,
                )

                survivor2 = self.compete(
                    parent2,
                    child2,
                )

            else:

                survivor1 = self.compete(
                    parent1,
                    child2,
                )

                survivor2 = self.compete(
                    parent2,
                    child1,
                )

            new_population.append(survivor1)
            new_population.append(survivor2)

        # Preserve the remaining individual if
        # the population size is odd.
        if self.population_size % 2 != 0:
            new_population.append(
                shuffled_population[-1]
            )

        self.population = np.asarray(
            new_population,
            dtype=float,
        )


if __name__ == "__main__":

    for benchmark in ["M1", "M4"]:

        ga = DeterministicCrowding(
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