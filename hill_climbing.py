"""Random-Restart Hill Climbing implementation."""

from experiment import run_experiment
import numpy as np

BENCHMARKS_TO_RUN = ["sphere", "rastrigin", "rosenbrock"]
SEEDS = [0, 1, 2, 3, 4]
MAX_EVALUATIONS = 20_000

# Design choices:
# - step_size=0.3: Gaussian perturbation scale balancing local exploration
#   vs. precision. Adapted to the ~10-unit domain widths.
# - local_budget_fraction=0.05: each restart gets 5% of budget (1000 evals),
#   enough to converge a local basin without over-committing to one region.
PARAMETERS = {
    "step_size": 0.3,
    "local_budget_fraction": 0.05,
}


def hill_climbing(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    step_size=0.3,
    local_budget_fraction=0.05,
    **parameters,
):
    """
    Random-Restart Hill Climbing with Gaussian neighborhood.

    Design rationale:
    - Neighborhood: additive Gaussian noise (std=step_size). Continuous,
      isotropic perturbations work across smooth and multimodal landscapes.
    - Local budget: fixed fraction per restart forces ~20 restarts over total
      budget, preventing stagnation at poor local optima (crucial for Rastrigin).
    - Restart policy: uniform random re-initialization maintains global coverage.
    - Boundary: clip after perturbation — simple and domain-safe.
    - Acceptance: greedy (strict improvement). Fast local exploitation;
      restarts supply global diversity.
    """
    local_budget = max(1, int(max_evaluations * local_budget_fraction))

    current = rng.uniform(lower_bound, upper_bound, size=dimension)
    current_value = objective(current)
    local_evals = 1

    while objective.remaining > 0:
        if local_evals >= local_budget:
            current = rng.uniform(lower_bound, upper_bound, size=dimension)
            current_value = objective(current)
            local_evals = 1
            if objective.remaining == 0:
                break
            continue

        neighbor = current + rng.normal(0, step_size, size=dimension)
        neighbor = np.clip(neighbor, lower_bound, upper_bound)
        neighbor_value = objective(neighbor)
        local_evals += 1

        if neighbor_value < current_value:
            current = neighbor
            current_value = neighbor_value


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            hill_climbing,
            benchmark_name,
            list(SEEDS),
            PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value:.6g}")


if __name__ == "__main__":
    main()
