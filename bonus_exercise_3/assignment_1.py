import time

import chaospy as cp
import numpy as np

from utils.sobol import monte_carlo_sobol, pseudo_spectral_sobol


def get_distribution(
    c_lims: tuple[float, float],
    k_lims: tuple[float, float],
    f_lims: tuple[float, float],
    y0_lims: tuple[float, float],
    y1_lims: tuple[float, float],
) -> cp.Distribution:
    """Creates the joint distribution over the stochastic parameters."""

    # TODO: create a joint distribution over the stochastic parameters.
    distribution = cp.J(
        cp.Uniform(c_lims[0], c_lims[1]),
        cp.Uniform(k_lims[0], k_lims[1]),
        cp.Uniform(f_lims[0], f_lims[1]),
        cp.Uniform(y0_lims[0], y0_lims[1]),
        cp.Uniform(y1_lims[0], y1_lims[1]),
    )


    return distribution


def run_method(method, **kwargs):
    """Runs the specified method and prints the results.

    The results include the first and total order Sobol' indices as well as
    the elapsed time to run the method."""

    # TODO: run the method and print the results.
    start = time.time()
    first_order, total_order = method(**kwargs)
    elapsed = time.time() - start

    print(f"Elapsed time: {elapsed:.3f}s")
    print("First order Sobol indices")
    print(first_order)
    print("Total order Sobol indices")
    print(total_order)

    return first_order, total_order, elapsed


if __name__ == "__main__":
    # TODO: set the stochastic parameters.
    c_lims = [0.08, 0.12]
    k_lims = [0.03, 0.04]
    f_lims = [0.08, 0.12]
    y0_lims = [0.45, 0.55]
    y1_lims = [-0.05, 0.05]


    # TODO: set the determinisic parameters.
    fixed_args = {"omega": 1.0}

    # TODO: set the parameters of the methods.
    quadrature_degrees = [3, 4]
    pce_degrees = [3, 4]
    # Match the full tensor-product PCE grids K^5 as requested in the hint.
    n_samples_list = [243, 1024]

    # TODO: set the time domain
    T_max = 10
    dt = 0.01
    t_grid = np.arange(0, T_max + dt, dt)

    ###########################################################################

    # TODO: define the distribution over the stochastic parameters.
    distribution = get_distribution(c_lims, k_lims, f_lims, y0_lims, y1_lims)

    # TODO: run the pseudo-spectral method on full grid.

    # TODO: run the pseudo-spectral method on sparse grid.

    # TODO: run the Monte Carlo method.

    for K, N, n_samples in zip(quadrature_degrees, pce_degrees, n_samples_list):
        quadrature_degree = K - 1
        pce_degree = N - 1

        print(f"\nK={K}, N={N}, Monte Carlo samples={n_samples}")

        print("\nPseudo-spectral full grid")
        run_method(
            pseudo_spectral_sobol,
            pce_degree=pce_degree,
            quadrature_degree=quadrature_degree,
            distribution=distribution,
            t_grid=t_grid,
            fixed_args=fixed_args,
            sparse=False,
        )

        print("\nPseudo-spectral sparse grid")
        run_method(
            pseudo_spectral_sobol,
            pce_degree=pce_degree,
            quadrature_degree=quadrature_degree,
            distribution=distribution,
            t_grid=t_grid,
            fixed_args=fixed_args,
            sparse=True,
        )

        print("\nMonte Carlo Saltelli/Jansen")
        run_method(
            monte_carlo_sobol,
            n_samples=n_samples,
            distribution=distribution,
            t_grid=t_grid,
            fixed_args=fixed_args,
        )

    ###########################################################################
