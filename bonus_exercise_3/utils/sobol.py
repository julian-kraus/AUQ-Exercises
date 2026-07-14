import chaospy as cp
import numpy as np
import numpy.typing as npt

from .oscillator import Oscillator


def _evaluate_oscillator(
    samples: npt.NDArray, t_grid: npt.NDArray, fixed_args: dict[str, float]
) -> npt.NDArray:
    """Evaluates the oscillator model for given samples."""

    outputs = np.zeros(samples.shape[0])

    for i, sample in enumerate(samples):
        c, k, f, y0, y1 = sample
        oscillator = Oscillator(c=c, k=k, f=f, omega=fixed_args["omega"])
        positions = oscillator.discretize("odeint", y0, y1, t_grid)
        Y = positions[-1]
        outputs[i] = Y

    return outputs


def monte_carlo_sobol(
    n_samples: int,
    distribution: cp.Distribution,
    t_grid: npt.NDArray[np.float64],
    fixed_args: dict[str, float],
) -> tuple[float, float]:
    """Computes the Sobol' indices using Monte Carlo sampling."""
    dimension = len(distribution)
    
    unit_samples = cp.create_sobol_samples(n_samples, 2 * dimension).T

    A = distribution.inv(unit_samples[:, :dimension].T).T
    B = distribution.inv(unit_samples[:, dimension:].T).T

    first_order = np.zeros(A.shape[1])
    total_order = np.zeros(A.shape[1])

    y_A = _evaluate_oscillator(A, t_grid, fixed_args)
    y_B = _evaluate_oscillator(B, t_grid, fixed_args)
    variance = np.var(np.concatenate([y_A, y_B]), ddof=1)

    for i in range(A.shape[1]):
        A_B_i = A.copy()
        A_B_i[:, i] = B[:, i]

        y_AB_i = _evaluate_oscillator(A_B_i, t_grid, fixed_args)

        S_i = np.mean(y_B * (y_AB_i - y_A)) / variance
        ST_i = np.mean((y_A - y_AB_i) ** 2) / (2 * variance)

        first_order[i] = S_i
        total_order[i] = ST_i

    return first_order, total_order


def pseudo_spectral_sobol(
    pce_degree: int,
    quadrature_degree: int,
    distribution: cp.Distribution,
    t_grid: npt.NDArray[np.float64],
    fixed_args: dict[str, float],
    sparse=True,
) -> tuple[float, float]:
    """Computes the Sobol' indices using a pseudo-spectral method."""
    orthogonal_expansion = cp.generate_expansion(pce_degree, distribution)

    nodes, weights = cp.generate_quadrature(
        quadrature_degree,
        distribution,
        rule="gaussian",
        sparse=sparse,
    )

    samples = nodes.T
    evaluations = _evaluate_oscillator(samples, t_grid, fixed_args)

    pce = cp.fit_quadrature(
        orthogonal_expansion,
        nodes,
        weights,
        evaluations,
    )

    first_order = cp.Sens_m(pce, distribution)
    total_order = cp.Sens_t(pce, distribution)

    return first_order, total_order
