import os
import time
from collections import defaultdict

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.helpers import compute_errors, generate_grid, load_reference, simulate
from utils.interpolation import FirstBarycentricLagrange
from utils.oscillator import Oscillator


def estimate_monte_carlo(
    omega_distr: cp.Distribution,
    n_samples: int,
    target_t: float,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
) -> npt.NDArray:
    # TODO-DONE: return the trajectories at the target time index for the given number of samples.
    # ====================================================================
    # solutions = np.empty(n_samples) #placeholder

    #Interested only in solutions for the target time, in the task it's t=10
    time_grid = np.arange(0, target_t + 0.01, 0.01)

    #Fixed the seed, to get reproducible results
    omega_samples = omega_distr.sample(size=np.array([n_samples]), rule="random", seed=42)
    solutions_mc = simulate(t_grid=time_grid,
                         omega_samples=omega_samples,
                         model_kwargs=model_kwargs,
                         init_cond=init_cond
                         )
    #Return the values for target time only,so index -1
    solutions = solutions_mc[:,-1]

    # ====================================================================
    return solutions


def fit_lagrange(
    omega_bounds: tuple[float, float],
    n_nodes: int,
    target_t: float,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
) -> FirstBarycentricLagrange:
    # TODO-DONE: Fit the Lagrange interpolator.
    # ====================================================================
    # interpolator = None #placeholder

    # The interpolation points grid is generated with a helper function
    # Using Chebyshev grid, as per the hint
    grid = generate_grid(bounds=omega_bounds, n_nodes=n_nodes, grid_type="chebyshev")

    # The values are the evaluations at the interpolation points
    values = np.empty(n_nodes)

    # To calculate the values for the interpolation points, we need to use the discretization of the oscillator

    # The deterministic oscillator parameters are:
    c = model_kwargs["c"]
    k = model_kwargs["k"]
    f = model_kwargs["f"]
    y0 = init_cond["y0"]
    y1 = init_cond["y1"]
    method = "odeint" # it's faster than Euler for larger sample sizes as seen in the tutorial

    time_grid = np.arange(0, target_t + 0.01, 0.01)

    for _i in range(n_nodes):
        oscillator = Oscillator(c=c, k=k, f=f, omega=grid[_i])
        #Interested only in the first component for the target time t=10, so index -1
        values[_i] = oscillator.discretize(y0=y0, y1=y1, method=method, t_grid=time_grid)[-1]

    # Fit the interpolator
    interpolator = FirstBarycentricLagrange(nodes=grid, values=values)
    interpolator.fit()
    # ====================================================================
    return interpolator
    


def evaluate_pce(
    interpolator: FirstBarycentricLagrange, omega_distr: cp.Distribution, n_samples: int
) -> npt.NDArray:
    # TODO-DONE: compute the trajectories using the Lagrange surrogate.
    # ====================================================================
    # solutions = np.zeros(n_samples) #placeholder

    #Fixed the seed, to get reproducible results
    omega_samples = omega_distr.sample(size=np.array([n_samples]), rule="random", seed=42)
    solutions = interpolator.evaluate(omega_samples)

    # ====================================================================
    return solutions


if __name__ == "__main__":
    # TODO-DONE: define the parameters of the simulations.
    # ====================================================================
    #pass #placeholder

    #Reusing adapted code from Bonus exercise 1, Task 4

    # Oscillator parameters
    # The deterministic oscillator parameters are:
    # c = 0.5, k = 2.0, f = 0.5
    model_kwargs = defaultdict(float)
    model_kwargs["c"] = 0.5
    model_kwargs["k"] = 2.0
    model_kwargs["f"] = 0.5
    # y0 = 0.5, y1 = 0.0
    init_cond = defaultdict(float)
    init_cond["y0"] = 0.5
    init_cond["y1"] = 0.0
    # The stochastic oscillator parameter is omega ~ U(0.95, 1.05)
    omega_distr = cp.Uniform(0.95, 1.05)
    omega_bounds = (0.95, 1.05)

    # Time points from 0 to 10 with t_delta = 0.01
    # Arrange creates values in half-open interval [start, stop),
    # hence padding the stop with an additional step
    t_grid = np.arange(0, 10.01, 0.01)

    # Time point of interest is y_0(10)
    t_target = 10

    # Number of interpolation points:
    N = [2, 5, 10, 20]

    # Number of Monte Carlo samples
    M = [10, 100, 1000, 10000]

    # Get the reference values with the helper method
    mean_ref, var_ref = load_reference("oscillator_ref.txt")
    # ====================================================================

    # TODO-DONE: perform PCE computations and record the time taken.
    # ====================================================================
    #pass #placeholder

    results_pce = defaultdict(lambda: defaultdict(lambda: np.ndarray(0)))
    times_pce = defaultdict(lambda: defaultdict(float))


    for n in N:
        start_pce_iter = time.time()
        # Build the |N| Lagrange interpolants
        interpolator = fit_lagrange(omega_bounds=omega_bounds,
                                    n_nodes=n,
                                    target_t=t_target,
                                    model_kwargs=model_kwargs,
                                    init_cond=init_cond)
        end_pce_iter = time.time()
        # Compute for the M samples
        for m in M:
            start_pce_eval = time.time()
            results_pce[n][m] = evaluate_pce(interpolator, omega_distr, n_samples=m)
            end_pce_eval = time.time()
            times_pce[n][m] = end_pce_iter - start_pce_iter + end_pce_eval - start_pce_eval
    # ====================================================================

    # TODO-DONE: perform Monte Carlo sampling and record the time taken.
    # ====================================================================
    #pass #placeholder

    results_mc = defaultdict(lambda: np.ndarray(0))
    times_mc = defaultdict(float)

    # directly using Monte Carlo, as in bonus exercise 1
    for m in M:
        start_mc = time.time()
        results_mc[m] = estimate_monte_carlo(omega_distr=omega_distr,
                                             n_samples=m,
                                             target_t=t_target,
                                             model_kwargs=model_kwargs,
                                             init_cond=init_cond)
        end_mc = time.time()
        times_mc[m] = end_mc - start_mc

    # ====================================================================

    # TODO-DONE: compute relative errors.
    # ====================================================================
    #pass #placeholder

    mean_errors_pce = np.empty((len(N), len(M)))
    var_errors_pce = np.empty((len(N), len(M)))
    mean_errors_mc = np.empty(len(M))
    var_errors_mc = np.empty(len(M))

    _i = 0
    _j = 0
    for n in N:
        for m in M:
            error_tuple = compute_errors(samples=results_pce[n][m], mean_ref=mean_ref, var_ref=var_ref)
            mean_errors_pce[_i][_j] = error_tuple[0]
            var_errors_pce[_i][_j] = error_tuple[1]
            _j+=1
        _i+=1
        _j = 0

    _i = 0
    for m in M:
        error_tuple = compute_errors(samples=results_mc[m], mean_ref=mean_ref, var_ref=var_ref)
        mean_errors_mc[_i] = error_tuple[0]
        var_errors_mc[_i] = error_tuple[1]
        _i+=1
    # ====================================================================

    # TODO-DONE: plot the results.
    # You may reuse code from previous exercises.
    # ====================================================================
    # pass # placeholder

    #Plotting relative error of the mean over number of samples
    fig1, plot1 = plt.subplots()
    plot1.set_title("Mean Relative Err")
    plot1.set_xlabel("Number of samples")
    plot1.set_ylabel("Mean Rel Error")
    plot1.loglog(M, mean_errors_mc, label="Monte Carlo", marker='o')
    linestyles = ["solid", "dashed" , "dotted", (0,(1,10))]
    for i, n in enumerate(N):
        plot1.loglog(M, mean_errors_pce[i,:], label="Lagrange for n="+repr(n), marker='o', linestyle=linestyles[i])
    plot1.legend()
    fig1.tight_layout()
    fig1.savefig("a1_mean_rel_err_plot.png")

    #Plotting relative error of the variance over number of samples
    fig2, axes2 = plt.subplots()
    axes2.set_title("Variance Relative Err")
    axes2.set_xlabel("Number of samples")
    axes2.set_ylabel("Var Rel Error")
    axes2.loglog(M, var_errors_mc, label="Monte Carlo", marker='o')
    linestyles = ["solid", "dashed" , "dotted", (0,(1,10))]
    for i, n in enumerate(N):
        axes2.loglog(M, var_errors_pce[i,:], label="Lagrange for n="+repr(n), marker='o', linestyle=linestyles[i])
    axes2.legend()
    fig2.tight_layout()
    fig2.savefig("a1_var_rel_err_plot.png")

    #Printing the timing results
    print("Times Monte Carlo: ")
    print("| ",end="")
    for m in M:
        print(times_mc[m],end=" | ")

    for n in N:
        print("\nTimes Lagrange for n="+repr(n))
        print("| ",end="")
        for m in M:
            print(times_pce[n][m],end=" | ")

    # ====================================================================
