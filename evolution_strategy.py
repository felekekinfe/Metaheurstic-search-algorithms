"""Evolution Strategy (μ, λ) with self-adaptive step sizes."""

from experiment import run_experiment
import numpy as np

BENCHMARKS_TO_RUN = ["sphere", "rastrigin", "rosenbrock"]
SEEDS = [0, 1, 2, 3, 4]
MAX_EVALUATIONS = 20_000

# Design choices (justified in docstring):
# mu=15, lam=100: ratio lam/mu ≈ 7, standard for (mu,lam)-ES.
# sigma=0.5: initial step size; self-adaptation refines this per-individual.
# (mu, lambda) replacement: offspring-only survivor selection encourages
#   leaving poor basins (better for multimodal Rastrigin).
PARAMETERS = {
    "mu": 15,
    "lam": 100,
    "sigma_init": 0.5,
}


def evolution_strategy(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    mu=15,
    lam=100,
    sigma_init=0.5,
    **parameters,
):
    """
    (μ, λ)-ES with per-individual self-adaptive step sizes.

    Design rationale:
    - (μ, λ) replacement: survivors are drawn from offspring only, not parents.
      This allows the population to abandon poor local optima, critical for
      Rastrigin. (μ + λ) would be elitist but can trap the population.
    - μ=15, λ=100: λ/μ ≈ 7, a well-established ratio. Large λ increases
      coverage; μ=15 provides stable selection pressure.
    - Self-adaptive σ per individual: each individual carries its own step size,
      updated via log-normal mutation. Learning rate τ = 1/sqrt(2*sqrt(d))
      (Schwefel's rule). This adapts exploitation to local landscape curvature.
    - σ_min clamp: prevents step size collapsing to 0 (premature convergence).
    - Truncation selection: top-μ by fitness. Simple, effective, well-understood.
    - Boundary: clip offspring to domain. No bounce needed; Gaussian rarely
      overshoots far with self-adapted small σ.
    - Budget accounting: population initialization (mu evals) + lam per generation.
      We run as many full generations as fit in the budget.
    """
    tau = 1.0 / np.sqrt(2.0 * np.sqrt(dimension))   # per-individual learning rate
    sigma_min = 1e-5

    # Initialize parent population: (individual, sigma, fitness)
    parents = []
    for _ in range(mu):
        if objective.remaining == 0:
            break
        x = rng.uniform(lower_bound, upper_bound, size=dimension)
        f = objective(x)
        parents.append((x, sigma_init, f))

    if not parents:
        return

    while objective.remaining >= lam:
        offspring = []

        for _ in range(lam):
            # Pick a random parent
            idx = rng.integers(0, len(parents))
            px, ps, _ = parents[idx]

            # Self-adapt step size: log-normal mutation
            new_sigma = max(sigma_min, ps * np.exp(tau * rng.standard_normal()))

            # Mutate individual
            child = px + new_sigma * rng.standard_normal(size=dimension)
            child = np.clip(child, lower_bound, upper_bound)

            f = objective(child)
            offspring.append((child, new_sigma, f))

        # (μ, λ) truncation: best μ from offspring only
        offspring.sort(key=lambda t: t[2])
        parents = offspring[:mu]

    # Use remaining budget one-by-one (fractional generation)
    while objective.remaining > 0:
        idx = rng.integers(0, len(parents))
        px, ps, _ = parents[idx]
        new_sigma = max(sigma_min, ps * np.exp(tau * rng.standard_normal()))
        child = px + new_sigma * rng.standard_normal(size=dimension)
        child = np.clip(child, lower_bound, upper_bound)
        f = objective(child)
        # Update parent if improved (mini elitism for remainder)
        if f < parents[idx][2]:
            parents[idx] = (child, new_sigma, f)


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            evolution_strategy,
            benchmark_name,
            list(SEEDS),
            PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value:.6g}")


if __name__ == "__main__":
    main()
