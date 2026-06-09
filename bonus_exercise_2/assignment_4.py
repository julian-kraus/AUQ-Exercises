import chaospy as cp
import matplotlib.pyplot as plt
import numpoly
import numpy as np
import numpy.typing as npt

from utils.helpers import load_reference, simulate, relative_error


def compute_coefficients(
    nodes: npt.NDArray,
    weights: npt.NDArray,
    polynomials: numpoly.ndpoly,
    target_t: float,
    model_kwargs: dict,
    init_cond: dict,
    mode: str,
) -> npt.NDArray:
    # TODO - Done: compute the coefficients using the quadrature rule, either manually.
    # or using chaospy functionality.
    # ====================================================================
    coefficients = np.zeros(polynomials.shape[0])

    time_grid = np.arange(0, target_t + 0.01, 0.01)
    sample_solutions = simulate(
        t_grid=time_grid,
        omega_samples=nodes[0],
        model_kwargs=model_kwargs,
        init_cond=init_cond,
    )
    solves = sample_solutions[:, -1]

    match mode:
        case "manual":
            K = len(polynomials)

            coefficients = np.array([
                np.sum(solves * polynomials[i](nodes[0]) * weights)
                for i in range(K)
            ])
        case "chaospy":
            pce, fourier_coeffs = cp.fit_quadrature(
                polynomials,
                nodes,
                weights,
                solves,
                retall=1
            )
            coefficients = fourier_coeffs
        case _:
            print("Unknown mode")

    # ====================================================================
    return coefficients


def compute_moments(coefficients: npt.NDArray) -> tuple[float, float]:
    # TODO - Done: compute the target mean and variance from the PCE coefficients.
    # ====================================================================
    # The mean is the first coefficient
    mean = coefficients[0]
    # The variance is the sum of the coefficient with index > 0 squared
    variance = np.sum(coefficients[1:] ** 2)

    # ====================================================================
    return mean, variance

#--------------------------------
# Plotting helper
def plot_error_subplot(
    ax,
    x_values,
    manual_errors,
    chaospy_errors,
    title,
):
    ax.plot(
        x_values,
        manual_errors,
        marker="o",
        label="Manual",
    )

    ax.plot(
        x_values,
        chaospy_errors,
        marker="x",
        label="Chaospy",
    )

    for x, y in zip(x_values, manual_errors):
        ax.annotate(
            f"{y:.2e}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )

    for x, y in zip(x_values, chaospy_errors):
        ax.annotate(
            f"{y:.2e}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
        )

    ax.set_xlabel("N = K")
    ax.set_ylabel("Relative error")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

#--------------------------------

if __name__ == "__main__":
    # TODO - Done: define the parameters of the simulations.
    # ====================================================================

    # Fetching MC reference values
    mean_mc, variance_mc = load_reference("oscillator_ref.txt")

    # Defining model (oscillator) params and initial condition according to the problem statement
    model_kwargs = {
        "c": 0.5,
        "k": 2.0,
        "f": 0.5,
    }

    init_cond = {
        "y0": 0.5,
        "y1": 0.0,
    }

    # Set target t = 10
    target_t = 10

    # Set N = [1,2,3,4,5,6]
    N_range = range(1, 7)

    # Saving the manual and chaospy coefficients for comparison
    coefficients_manual = []
    coefficients_chaospy = []

    # Saving chaospy relative error for mean and var
    mean_chaospy_error = []
    variance_chaospy_error = []

    # Saving manual relative error for mean and var
    mean_manual_error = []
    variance_manual_error = []

    # Defining the distribution of omega
    distribution = cp.Uniform(0.95,1.05)


    for N in N_range:
        K = N

        # Using Gaussian rule due to uniformly distr. stochastic variable
        nodes, weights = cp.generate_quadrature(
            K-1,
            distribution,
            rule="gaussian"
        )

        # Ensures orthonormal polynomials
        polynomials = cp.generate_expansion(
            N-1,
            distribution,
            normed=True
        )

    # ====================================================================

    # TODO - Done: compute pseudo-spectral coefficients.
    # ====================================================================

        coefficients_m = compute_coefficients(
            nodes,
            weights,
            polynomials,
            target_t,
            model_kwargs,
            init_cond,
            mode="manual"
        )

        coefficients_c = compute_coefficients(
            nodes,
            weights,
            polynomials,
            target_t,
            model_kwargs,
            init_cond,
            mode="chaospy"
        )

        coefficients_manual.append(coefficients_m)
        coefficients_chaospy.append(coefficients_c)
    # ====================================================================

    # TODO - Done: compute the moments and calculate their errors.
    # ====================================================================

        mean, variance = compute_moments(coefficients_m)

        mean_manual_error.append(relative_error(mean, mean_mc))
        variance_manual_error.append(relative_error(variance, variance_mc))

        mean, variance = compute_moments(coefficients_c)

        mean_chaospy_error.append(relative_error(mean, mean_mc))
        variance_chaospy_error.append(relative_error(variance, variance_mc))


    # ====================================================================

    # TODO - Done: plot the results.
    # You may reuse code from previous exercises.
    # ====================================================================
    # Plotting coefficients for N = 6
    coeff_manual_final = coefficients_manual[-1]
    coeff_chaospy_final = coefficients_chaospy[-1]

    print("\nCoefficient comparison (N = K = 6)")
    print("-" * 80)
    print(f"{'i':<5} {'Manual':<18} {'Chaospy':<18} {'|Difference|':<18}")
    print("-" * 80)

    for i, (cm, cc) in enumerate(zip(coeff_manual_final, coeff_chaospy_final)):
        diff = abs(cm - cc)
        print(
            f"{i:<5} "
            f"{cm:<18.10e} "
            f"{cc:<18.10e} "
            f"{diff:<18.10e}"
        )

    # Plot relative mean and variance errors for manual and chaospy approaches
    N_values = list(N_range)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_error_subplot(
        axes[0],
        N_values,
        mean_manual_error,
        mean_chaospy_error,
        "Mean error",
    )

    plot_error_subplot(
        axes[1],
        N_values,
        variance_manual_error,
        variance_chaospy_error,
        "Variance error",
    )

    fig.suptitle("Relative errors of PCE estimates")

    plt.tight_layout()
    plt.savefig("plots/relative_errors.png")
    plt.show()


    # ====================================================================
