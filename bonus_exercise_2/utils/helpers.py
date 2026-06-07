import numpy as np
import numpy.typing as npt
import scipy.special as sp

from utils.oscillator import Oscillator


def generate_grid(
    bounds: tuple[float, float], n_nodes: int, grid_type: str = "uniform"
) -> npt.NDArray:
    # TODO-DONE: generate the interpoliation grid on [low, high] with n_nodes points.
    # You may reuse code from previous exercises.
    # ====================================================================
    # grid = np.zeros(n_nodes) # placeholder

    # Reusing the code from Exercise 5, as allowed above
    low, high = bounds
    match grid_type:
        case "uniform":
            grid = np.linspace(low, high, n_nodes)
        case "chebyshev":
            nodes, _ = sp.roots_chebyc(n_nodes)
            # roots_chebyc returns the nodes in the interval [-2, 2].
            # We need to scale them to be in [low, high].
            grid = (nodes + 2) / 4 * (high - low) + low
        case _:
            raise ValueError(f"Unknown grid type: {grid_type}.")
    # ====================================================================
    return grid


def compute_errors(
    samples: npt.NDArray, mean_ref: float, var_ref: float
) -> tuple[float, float]:
    # TODO-DONE: compute the relative errors of the mean and variance
    # estimates.
    # You may reuse code from previous exercises.
    # ====================================================================
    # mean_error, var_error = 0.0, 0.0 #placeholder

    # Reusing the code from Bonus Exercise 1, Task 4, as allowed above:

    # Using numpy, some quick statistics
    mean_of_estimation = np.mean(samples, dtype=float)
    # Taking the unbiased estimator as in the tutor exercises (normalization with 1/n-1)
    var_of_estimation = np.var(samples, ddof=1, dtype=float)

    # Using the relative error formula from the task sheet
    mean_error = np.abs(1-mean_of_estimation/mean_ref)
    var_error = np.abs(1-var_of_estimation/var_ref)
    # ====================================================================
    return mean_error, var_error


def load_reference(filename: str) -> tuple[float, float]:
    # TODO-DONE: load reference values for the mean and variance.
    # You may reuse code from previous exercises.
    # ====================================================================
    #mean, var = 0.0, 0.0 #placeholder

    # Reusing the code from Bonus Exercise 1, Task 4, as allowed above:

    # open the file with the two references in read mode, use "with" for auto closing of file
    with open(filename, "r") as file_with_refs:
        # read the two lines, convert from string to float and the values in the (mean,variance) tuple
        mean, var = float(file_with_refs.readline()), float(file_with_refs.readline())
    # ====================================================================
    return mean, var


def simulate(
    t_grid: npt.NDArray,
    omega_samples: npt.NDArray,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
) -> npt.NDArray:
    # TODO-DONE: simulate the oscillator with the given parameters and return
    # generated solutions.
    # ====================================================================
    # sample_solutions = np.zeros_like(t_grid) #placeholder

    # Reusing adapted code from Bonus exercise 1, Task 4

    # The deterministic oscillator parameters are:
    c = model_kwargs["c"]
    k = model_kwargs["k"]
    f = model_kwargs["f"]
    y0 = init_cond["y0"]
    y1 = init_cond["y1"]
    method = "odeint" # it's faster than Euler for larger sample sizes as seen in the tutorial

    n_samples = len(omega_samples)

    # Return data structure: a 2D matrix
    # Each of the rows corresponds to a simulated oscillator for a sample omega_i in omega_samples
    # Each row contains an evaluated value for each of the time points in T_grid
    sample_solutions = np.empty((n_samples, len(t_grid)))

    # Simulating (evaluating) the oscillator N times
    for _i in range(n_samples):
        oscillator = Oscillator(c=c, k=k, f=f, omega=omega_samples[_i])
        sample_solutions[_i] = oscillator.discretize(method=method, y0=y0, y1=y1 , t_grid=t_grid)

    # ====================================================================
    return sample_solutions
