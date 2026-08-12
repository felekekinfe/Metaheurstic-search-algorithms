"""Genetic Algorithm with tournament selection, blend crossover, Gaussian mutation."""

from experiment import run_experiment
import numpy as np

BENCHMARKS_TO_RUN = ["sphere", "rastrigin", "rosenbrock"]
SEEDS = [0, 1, 2, 3, 4]
MAX_EVALUATIONS = 20_000

# Design choices (justified in docstring):
# pop_size=80: generation cost=80 evals → ~250 generations over 20k budget.
# tournament_k=3: moderate selection pressure; k=2 too weak, k=5 too greedy.
# crossover_prob=0.9, alpha=0.5: blend crossover with high frequency builds
#   good building blocks from two fit parents.
# mutation_prob=0.1: per-gene Gaussian mutation; 10% per gene in 10-D means
#   ~1 gene perturbed per child on average. Prevents premature convergence.
# mutation_scale=0.2: small Gaussian steps for fine-tuning.
PARAMETERS = {
    "pop_size": 80,
    "tournament_k": 3,
    "crossover_prob": 0.9,
    "alpha": 0.5,
    "mutation_prob": 0.1,
    "mutation_scale": 0.2,
    "elitism": 2,
}


def genetic_algorithm(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    pop_size=80,
    tournament_k=3,
    crossover_prob=0.9,
    alpha=0.5,
    mutation_prob=0.1,
    mutation_scale=0.2,
    elitism=2,
    **parameters,
):
    """
    Genetic Algorithm: tournament selection + blend crossover + Gaussian mutation.

    Design rationale:
    - Population size 80: balances diversity and per-generation cost.
      ~250 generations available — enough for convergence on smooth functions
      and partial convergence on Rosenbrock's narrow valley.
    - Tournament selection (k=3): deterministic tournament is parameter-free
      relative to roulette, works with negative/close fitness values, and
      gives controllable selection pressure. k=3 is a stable default.
    - Blend crossover BLX-α (α=0.5): samples child genes uniformly from the
      extended interval [min-α·gap, max+α·gap] between parents. Preserves
      intermediate values (good for Rosenbrock's correlated variables) and
      extends slightly beyond parents for exploration. Better than 1-point
      crossover on continuous problems.
    - Gaussian mutation (prob=0.1 per gene, scale=0.2): per-gene independent
      application means ~1 gene mutated per child in 10-D. Small scale for
      fine-tuning; probability ensures diversity without destroying good genes.
    - Elitism (top-2 carry over): prevents loss of the best solutions found.
      Elitism + (μ,λ)-style replacement gives the GA a safety net without
      stagnating like pure elitism.
    - Boundary: clip after crossover and mutation.
    """
    # ----- helpers -----
    def tournament_select(population, fitnesses):
        """Return index of winner from k random competitors."""
        idxs = rng.choice(len(population), size=tournament_k, replace=False)
        return idxs[np.argmin(fitnesses[idxs])]

    def blend_crossover(p1, p2):
        """BLX-alpha crossover: sample from extended interval between parents."""
        lo = np.minimum(p1, p2)
        hi = np.maximum(p1, p2)
        gap = hi - lo
        child = rng.uniform(lo - alpha * gap, hi + alpha * gap)
        return np.clip(child, lower_bound, upper_bound)

    def mutate(individual):
        mask = rng.random(dimension) < mutation_prob
        noise = rng.normal(0, mutation_scale, size=dimension)
        mutated = individual + mask * noise
        return np.clip(mutated, lower_bound, upper_bound)

    # ----- initialization -----
    population = []
    fitnesses_list = []
    for _ in range(pop_size):
        if objective.remaining == 0:
            break
        x = rng.uniform(lower_bound, upper_bound, size=dimension)
        f = objective(x)
        population.append(x)
        fitnesses_list.append(f)

    population = np.array(population)
    fitnesses = np.array(fitnesses_list)

    if len(population) == 0:
        return

    # ----- generational loop -----
    while objective.remaining >= pop_size:
        new_population = []
        new_fitnesses = []

        # Elitism: carry best individuals directly
        elite_idx = np.argsort(fitnesses)[:elitism]
        for idx in elite_idx:
            new_population.append(population[idx].copy())
            new_fitnesses.append(fitnesses[idx])

        # Fill rest with offspring
        while len(new_population) < pop_size:
            if objective.remaining == 0:
                break

            p1_idx = tournament_select(population, fitnesses)
            p2_idx = tournament_select(population, fitnesses)

            p1 = population[p1_idx]
            p2 = population[p2_idx]

            # Crossover
            if rng.random() < crossover_prob:
                child = blend_crossover(p1, p2)
            else:
                child = p1.copy()

            # Mutation
            child = mutate(child)

            f = objective(child)
            new_population.append(child)
            new_fitnesses.append(f)

        population = np.array(new_population)
        fitnesses = np.array(new_fitnesses)

    # ----- use remaining budget -----
    while objective.remaining > 0:
        p1_idx = tournament_select(population, fitnesses)
        p2_idx = tournament_select(population, fitnesses)
        child = blend_crossover(population[p1_idx], population[p2_idx])
        child = mutate(child)
        f = objective(child)
        # Replace worst if better
        worst_idx = np.argmax(fitnesses)
        if f < fitnesses[worst_idx]:
            population[worst_idx] = child
            fitnesses[worst_idx] = f


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            genetic_algorithm,
            benchmark_name,
            list(SEEDS),
            PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value:.6g}")


if __name__ == "__main__":
    main()
