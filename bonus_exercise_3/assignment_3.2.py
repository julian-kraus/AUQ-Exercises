import time
from collections import defaultdict
from functools import partial
from typing import Callable

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpoly import square

from utils.oscillator import Oscillator
from utils.wiener import WienerProcess


def generate_f_samples(
        mu: float,
        t_grid: npt.NDArray,
        n_samples: int,
        M: int | None,
        rng: np.random.Generator,
) -> list[Callable[[float], float]]:
    """Generates samples of the Wiener process."""

    # TODO-DONE: generate realizations of the Wiener process for f(t).
    # If M is None, we generate samples using the standard definition.
    # If M is not None, we generate samples using the KL expansion with M terms.
    # The samples are returned as a list of callable functions that
    # evaluate the Wiener process at a given time point.

    #placeholder return [lambda t: 0.0] * n_samples

    wiener = WienerProcess(mu=mu, t_grid=t_grid)
    # Generating Wiener process samples
    # (for now, just a list of values, we handle turning the samples into callable later)
    if M is None:
        # Generate samples using standard definition
        samples = wiener.generate(n_samples=n_samples, rng=rng)
    else:
        # Generate samples using the KL expansion with M terms
        samples = wiener.approximate_kl(n_samples=n_samples, M=M, rng=rng)

    # Turning the samples into a list of callable functions
    # that can be used to interpolate at t (given time point)
    callable_functions = [lambda t: 0.0] * n_samples
    for i in range(n_samples):
        callable_functions[i] = partial(np.interp, xp=t_grid, fp=samples[i])
    return callable_functions

def simulate(
        t_grid: npt.NDArray,
        f_samples: list[Callable[[float], float]],
        model_kwargs: dict[str, float],
        init_cond: dict[str, float],
) -> npt.NDArray:
    """Simulates the oscillator model for each sample of f(t)."""

    # TODO-DONE: simulate the oscillator model for each sample of f(t) and
    # return the trajectories as 2D array.

    #placeholder return np.zeros(len(f_samples), len(t_grid))
    # Reusing adapted code from Bonus exercise 2

    # The oscillator parameters are:
    c = model_kwargs["c"]
    k = model_kwargs["k"]
    omega = model_kwargs["omega"]
    y0 = init_cond["y0"]
    y1 = init_cond["y1"]
    method = "odeint" # it's faster than Euler for larger sample sizes as seen in the tutorial

    n_samples = len(f_samples)

    # Return data structure: a 2D matrix
    # Each of the rows corresponds to a simulated oscillator for a sample f in f_samples
    # Each row contains an evaluated value for each of the time points in T_grid
    sample_solutions = np.empty((n_samples, len(t_grid)))

    # Simulating (evaluating) the oscillator N times
    for _i in range(n_samples):
        oscillator = Oscillator(c=c, k=k, f=f_samples[_i], omega=omega)
        sample_solutions[_i] = oscillator.discretize(method=method, y0=y0, y1=y1 , t_grid=t_grid)

    return sample_solutions

def compute_metrics(solutions: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Computes the mean and standard deviation of the solutions."""

    # TODO-DONE: compute the metrics.
    #placeholder return np.zeros(solutions.shape[1]), np.zeros(solutions.shape[1])

    # The assignment sheet actually asks for mean and variance, not standard deviation
    # But variance is standard variation squared and this helper is already used with standard variance for the logic in the plot_solutions() helper

    mean = np.mean(solutions,axis=0)
    standard_deviation = np.std(solutions, axis=0, ddof=1)

    return mean, standard_deviation

def plot_solutions(
        t_grid: npt.NDArray, sampler_solutions: dict[str, npt.NDArray]
) -> plt.Figure:
    """Plots the oscillator trajectories for each sample of f."""
    n_plots = len(sampler_solutions)
    fig, axes = plt.subplots(
        1, n_plots, figsize=(6 * n_plots, 4), sharex=True, sharey=True
    )
    for ax, (name, solutions) in zip(axes, sampler_solutions.items()):
        mean, std = compute_metrics(solutions)
        ax.plot(t_grid, solutions.T, alpha=0.01, c="b")
        ax.plot(t_grid, mean, c="r", label="mean")
        ax.fill_between(
            t_grid, mean - std, mean + std, color="red", alpha=0.5, label="std"
        )

        # Add legend for samples manually.
        handles, _ = ax.get_legend_handles_labels()
        line = lines.Line2D([0], [0], color="b", label="Monte Carlo samples")
        handles.append(line)
        ax.legend(handles=handles)

        ax.set_title(name)
    return fig


if __name__ == "__main__":
    # TODO-DONE: set parameters of the model.
    #placeholder f_mean = None
    #placeholder model_kwargs = {"c": None, "k": None, "omega": None}
    #placeholder init_cond = {"y0": None, "y1": None}

    f_mean = 0.5
    model_kwargs = {"c": 0.5, "k": 2.0, "omega": 1.0}
    init_cond = {"y0": 0.5, "y1": 0}

    # TODO-DONE: set the time domain.
    #placeholder T_max = None
    #placeholder dt = None
    #placeholder t_grid = np.arange(0, T_max + dt, dt)

    T_max = 10
    dt = 0.01
    t_grid = np.arange(0, T_max + dt, dt)

    # TODO-DONE: set the number of Monte-Carlo samples and KL terms.
    #placeholder N = None
    #placeholder Ms = [None]
    #placeholder seed = None

    N = 1000
    Ms = [5,10,100]
    # Reuse seed as in the last bonus assignments
    seed = 42
    rng = np.random.default_rng(seed)

    ###########################################################################

    # TODO-DONE: generate samples of the Wiener process for f using the stadard
    # generation and the KL expansion for different M.

    WP_samples = defaultdict()
    WP_samples["standard WP"] = generate_f_samples(mu=f_mean, t_grid=t_grid, n_samples=N, M=None, rng=rng)
    for M in Ms:
        #Regenerate rng to make sure the same samples are used
        rng = np.random.default_rng(seed)
        WP_samples["KL expansion for M={}".format(M)] = generate_f_samples(mu=f_mean, t_grid=t_grid, n_samples=N, M=M, rng=rng)

    # TODO-DONE: simulate the oscillator model for each sample of f and record the
    # mean and standard deviation of the solutions at T_max.
    simulations = defaultdict()

    #Mean and variance are calculated by the plot helper provided in the template, but we need to print them as table too
    #First, calculate mean and standard deviation with the helper
    mean = defaultdict()
    standard_deviation = defaultdict()

    for name,f_samples in WP_samples.items():
        simulations[name] = simulate(t_grid=t_grid, f_samples=f_samples, model_kwargs=model_kwargs, init_cond=init_cond)
        mean[name],standard_deviation[name]=compute_metrics(simulations[name])

    # TODO-DONE: optionally, plot the solutions for each sample of f.
    #Plots with the helper
    fig = plot_solutions(t_grid=t_grid, sampler_solutions=simulations)
    fig.savefig("as3_2_simulators.png")

    #Printing mean, variance as requested by the assignment sheet
    print("Mean:")
    for name,value in mean.items():
        print(name+": {}".format(value[-1]))

    #print("Standard Deviation:")
    #for name,value in standard_deviation.items():
    #    print(name+": {}".format(value[-1]))

    # Using that variance is standard deviation squared
    print("Variance:")
    for name,value in standard_deviation.items():
        var = np.square(value[-1])
        print(name+": {}".format(var))
