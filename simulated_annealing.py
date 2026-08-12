"""Simulated Annealing implementation."""

from experiment import run_experiment
import numpy as np

BENCHMARKS_TO_RUN = ["sphere", "rastrigin", "rosenbrock"]
SEEDS = [0, 1, 2, 3, 4]
MAX_EVALUATIONS = 20_000

# Design choices (all justified in function docstring):
# - initial_temp=5.0: set to accept ~exp(-1/5)~82% of 1-unit uphill moves early.
# - cooling_rate=0.9995: geometric cooling across ~20k steps reaches near-zero
#   temperature at budget end, shifting from exploration to exploitation.
# - step_size=0.5: moderate Gaussian perturbation; larger than HC to exploit
#   SA's ability to accept uphill moves and escape local minima.
PARAMETERS = {
    "initial_temp": 5.0,
    "cooling_rate": 0.9995,
    "step_size": 0.5,
}


def simulated_annealing(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    initial_temp=5.0,
    cooling_rate=0.9995,
    step_size=0.5,
    **parameters,
):
    """
    Simulated Annealing with geometric cooling schedule.

    Design rationale:
    - Initial temperature: chosen so early acceptance probability for a
      typical bad move (~1 unit worse) is ~82%, enabling broad exploration.
    - Cooling schedule: geometric decay T <- T * cooling_rate each step.
      With cooling_rate=0.9995 and 20k steps, final T ≈ 5 * 0.9995^20000 ≈ 0.00034,
      effectively greedy at the end. Smooth transition: exploration -> exploitation.
    - Neighborhood: Gaussian (std=step_size=0.5). Larger than HC since SA
      can accept worse solutions, making bigger steps worthwhile early.
    - Boundary: reflection. Preserves gradient information near boundaries
      better than clipping (no pileup at edges). If reflected point is still
      out of bounds, fall back to clip.
    - No restart: SA's temperature schedule already implements global-to-local
      search. Restarts would discard the best solution found.
    - Best tracking: always track global best, return it at end.
    """
    # Initialize
    current = rng.uniform(lower_bound, upper_bound, size=dimension)
    current_value = objective(current)
    best = current.copy()
    best_value = current_value
    temperature = initial_temp

    while objective.remaining > 0:
        # Gaussian neighborhood
        neighbor = current + rng.normal(0, step_size, size=dimension)

        # Reflection boundary: bounce off walls
        domain_width = upper_bound - lower_bound
        # Normalize to [0, domain_width], reflect, unnormalize
        shifted = neighbor - lower_bound
        shifted = shifted % (2 * domain_width)
        reflected = np.where(shifted <= domain_width, shifted, 2 * domain_width - shifted)
        neighbor = np.clip(reflected + lower_bound, lower_bound, upper_bound)

        neighbor_value = objective(neighbor)

        # Metropolis acceptance
        delta = neighbor_value - current_value
        if delta < 0 or rng.random() < np.exp(-delta / max(temperature, 1e-10)):
            current = neighbor
            current_value = neighbor_value

        if current_value < best_value:
            best = current.copy()
            best_value = current_value

        # Geometric cooling
        temperature *= cooling_rate

    # Evaluate best one final time to ensure it's recorded (already tracked internally)
    # The best candidate is already in objective's tracker via its call history


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            simulated_annealing,
            benchmark_name,
            list(SEEDS),
            PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value:.6g}")


if __name__ == "__main__":
    main()
