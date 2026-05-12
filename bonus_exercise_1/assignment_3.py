import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.sampling import control_variates, importance_sampling, monte_carlo


def f(x: npt.NDArray) -> npt.NDArray:
    # TODO: define the target function.
    # ====================================================================
    return np.exp(x)
    # ====================================================================


def analytical_integral() -> float:
    # TODO: compute the analytical integral of f on [0, 1].
    # ====================================================================
    return np.e - 1
    # ====================================================================


def run_monte_carlo(Ns: tuple[int, ...], seed: int = 42) -> list[float]:
    # TODO: run the Monte Carlo method and return the absolute error
    # of the estimation.
    # ====================================================================
    errors = []
    ground_truth = analytical_integral()
    for n in Ns:
        mean, rmse = monte_carlo(p=cp.Uniform(0, 1),
        n_samples=n,
        f=f,
        seed=seed
        )
        errors.append(np.absolute(mean - ground_truth).item())

    return errors
    # ====================================================================


def run_control_variates(
    Ns: tuple[int, ...], seed: int = 42
):
    # TODO: run the control variate method for and return the absolute
    # errors of the resulting estimations.
    # ====================================================================
    ground_truth = analytical_integral()
    cov_results = []
    covs = [
        (lambda x: x, 1 / 2),
        (lambda x: 1 + x, 3 / 2),
        (lambda x: 1 + x + x**2 / 2, 5 / 3),
    ]
    for phi, control_mean in covs:
        errors = []

        for n in Ns:
            mean = control_variates(p=cp.Uniform(0, 1),
            n_samples=n,
            f=f,
            phi=phi,
            control_mean=control_mean,
            seed=seed
            )
            errors.append(np.absolute(mean - ground_truth).item())
        cov_results.append(errors)

    return cov_results    
    # ====================================================================


def run_importance_sampling(
    Ns: tuple[int, ...], seed: int = 42
):
    # TODO: run the importance sampling method and return the absolute
    # errors of the resulting estimations.
    # ====================================================================
    ground_truth = analytical_integral()
    importance_results = []
    distributions = [
        cp.Beta(5, 1),
        cp.Beta(0.5, 0.5)
    ]
    for distribution in distributions:
        errors = []

        for n in Ns:
            mean = importance_sampling(
                p=cp.Uniform(0, 1),
                q=distribution,
                n_samples=n,
                f=f,
                seed=seed
            )
            errors.append(np.absolute(mean - ground_truth).item())
        importance_results.append(errors)

    return importance_results    
    # ====================================================================

def print_errors_pretty(Ns: tuple[int, ...], errors: list[float], name: str):
    print(f"Errors for {name}")
    for n, e in zip(Ns, errors):
        print(f"  N={n:>6}: {e:.6f}")



if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    Ns = (10, 100, 1000, 10000)
    seed = 42
    # ====================================================================

    # TODO: run all the methods.
    # ====================================================================
    mc_errors = run_monte_carlo(Ns, seed)
    print_errors_pretty(Ns, mc_errors, "Monte Carlo")


    cv_errors = run_control_variates(Ns, seed)
    cv_h1_errors = cv_errors[0]
    print_errors_pretty(Ns, cv_h1_errors, "Control Variate h1")
    cv_h2_errors = cv_errors[1]
    print_errors_pretty(Ns, cv_h2_errors, "Control Variate h2")
    cv_h3_errors = cv_errors[2]
    print_errors_pretty(Ns, cv_h3_errors, "Control Variate h3")


    is_errors = run_importance_sampling(Ns, seed)
    is_beta_5_1_errors = is_errors[0]
    print_errors_pretty(Ns, is_beta_5_1_errors, "Importance Sampling 5, 1")
    is_beta_05_05_errors = is_errors[1]
    print_errors_pretty(Ns, is_beta_05_05_errors, "Importance Sampling 0.5, 0.5")

    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(Ns, mc_errors, marker="o", linestyle="-", label="Monte Carlo")
    ax.loglog(Ns, cv_h1_errors, marker="s", linestyle="--", linewidth=3, label=r"CV: $h_1(x)=x$")
    ax.loglog(Ns, cv_h2_errors, marker="^", linestyle="--", label=r"CV: $h_2(x)=1+x$")
    ax.loglog(Ns, cv_h3_errors, marker="D", linestyle="--", label=r"CV: $h_3(x)=1+x+x^2/2$")
    ax.loglog(Ns, is_beta_5_1_errors, marker="*", linestyle=":", label=r"IS: $Beta(5,1)$")
    ax.loglog(Ns, is_beta_05_05_errors, marker="P", linestyle=":", label=r"IS: $Beta(0.5,0.5)$")

    ax.set_xlabel("Number of samples $N$")
    ax.set_ylabel("Absolute error")
    ax.set_title("Monte Carlo variance reduction comparison")
    ax.legend()
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.show()
    # ====================================================================
