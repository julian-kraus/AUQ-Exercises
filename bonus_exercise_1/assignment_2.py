from functools import partial
from typing import Callable

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.sampling import monte_carlo

Function = Callable[[npt.NDArray], npt.NDArray]


def f(x: npt.NDArray) -> npt.NDArray:
    # ✓ TODO: define the target function.
    # ====================================================================
    return np.sin(x)
    # ====================================================================


def analytical_integral(a: float, b: float) -> float:
    # ✓ TODO: compute the analytical integral of f on [a, b].
    # ====================================================================
    return -np.cos(b) + np.cos(a)
    # ====================================================================


def transform(samples: npt.NDArray, a: float, b: float) -> npt.NDArray:
    # ✓ TODO: implement the transformation of U from [0, 1] to [a, b].
    # ====================================================================
    samples = samples * (b - a) + a
    # ====================================================================
    return samples


def integrate_mc(
    f: Function,
    a: float,
    b: float,
    n_samples: int,
    with_transform: bool = False,
    seed: int = 42,
) -> tuple[float, float]:
    #  ✓ TODO: compute the integral with the Monta Carlo method.
    # Depending on 'with_transform', use the uniform distribution on [a, b]
    # directly or transform the uniform distribution on [0, 1] to [a, b].
    # Return the integral estimate and the corresponding rmse.
    # ====================================================================
    if with_transform:
        distr = cp.Uniform(0, 1)
        transform_ab = partial(transform, a=a, b=b)
    else:
        distr = cp.Uniform(a, b)
        transform_ab = None

    integral, rmse = monte_carlo(
        p=distr,
        n_samples=n_samples,
        f=f,
        transform=transform_ab,
        seed=seed,
    )

    return (b - a) * integral[0], (b - a) * rmse[0]


if __name__ == "__main__":

    # ✓ TODO: define the parameters of the simulation.
    # ====================================================================
    sample_sizes = [10, 100, 1000, 10000]
    seeds = [42, 43]

    # Assignment 2.1 =====================================================
    a = 0
    b = 1

    F = analytical_integral(a, b)

    for seed in seeds:
        exact_errors = []
        rmses = []

        for n_samples in sample_sizes:
            integral, rmse = integrate_mc(f, a, b, n_samples, seed=seed)
            exact_errors.append(np.abs(F - integral))
            rmses.append(rmse)

        plt.figure()

        plt.loglog(sample_sizes, exact_errors, marker="o", label="Exact error")
        plt.loglog(sample_sizes, rmses, marker="o", label="rmse")

        plt.title(f"Exact error vs RMSE (a=0, b=1, seed={seed})")
        plt.xlabel("Sample size")
        plt.ylabel("Error")
        plt.legend()
        plt.grid(True)

        plt.show()
        # plt.savefig(f"plots/assignment_2_1_seed{seed}.png")

    # Assignment 2.2 =====================================================
    a = 2
    b = 4

    F = analytical_integral(a, b)

    for seed in seeds:
        # With transform
        exact_errors_tr = []
        rmses_tr = []

        # Without transform
        exact_errors_no_tr = []
        rmses_no_tr = []

        for n_samples in sample_sizes:
            integral, rmse = integrate_mc(f, a, b, n_samples, seed=seed, with_transform=True)
            exact_errors_tr.append(np.abs(F - integral))
            rmses_tr.append(rmse)

        for n_samples in sample_sizes:
            integral, rmse = integrate_mc(f, a, b, n_samples, seed=seed, with_transform=False)
            exact_errors_no_tr.append(np.abs(F - integral))
            rmses_no_tr.append(rmse)

        plt.figure()

        plt.loglog(sample_sizes, exact_errors_tr, marker="o", label="Exact error (with transform)")
        plt.loglog(sample_sizes, rmses_tr, marker="o", label="RMSE (with transform)")
        plt.loglog(sample_sizes, exact_errors_no_tr, marker="x", label="Exact error (without transform)")
        plt.loglog(sample_sizes, rmses_no_tr, marker="x", label="RMSE (without transform)")

        plt.title(f"Exact error vs RMSE (a=2, b=4, seed={seed})")
        plt.xlabel("Sample size")
        plt.ylabel("Error")
        plt.legend()
        plt.grid(True)

        plt.show()
        # plt.savefig(f"plots/assignment_2_2_seed{seed}.png")


    # ====================================================================

