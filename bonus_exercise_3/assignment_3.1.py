import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.wiener import WienerProcess


def plot_eigenpairs(
    wiener: WienerProcess, n_terms: int, t_grid: npt.NDArray[np.float64]
) -> plt.Figure:
    """Plots the first n_terms eigenvalues and eigenfunctions of the Wiener process."""
    eigenvalues, eigenfunctions = wiener.kl_eigenpairs(n_terms)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.arange(1, n_terms + 1), eigenvalues, marker="o")
    axes[0].set_yscale("log")
    axes[0].set_title(f"First {n_terms} eigenvalues")
    axes[1].plot(t_grid, eigenfunctions(t_grid))
    axes[1].set_title(f"First {n_terms} eigenfunctions")
    return fig


if __name__ == "__main__":
    # ✓ TODO: set the configuration.
    T = 1 # Assume T + [0,1]
    n_points = 1000 # N
    t_grid = np.linspace(0, T, n_points)
    Ms = [10, 100, 1000]
    seed = 42
    n_samples = 1 # 1 realization of the process

    rng = np.random.default_rng(seed)
    # Initiate the Wiener Process object
    WP = WienerProcess(mu = 0, T=T, n_points=n_points)
    # ✓ TODO: generate one realization of the Wiener process using the
    # standard definition.
    standard_definition = WP.generate(n_samples, rng)

    # ✓ TODO: generate approximations of the Wiener process using the KL expansion.
    kl_expansions = {}
    for M in Ms:
        rng_M = np.random.default_rng(seed)
        kl_expansions[M] = WP.approximate_kl(n_samples, M, rng_M)

    # ✓ TODO: plot the approximation results.
    plt.figure(figsize=(10, 6))

    # Direct definition realization
    plt.plot(WP.t_grid, standard_definition[0], label="Direct definition", color="black", linewidth=1)

    # KL approximations for each M
    for M in Ms:
        plt.plot(WP.t_grid, kl_expansions[M][0], label=f"KL approximation, M={M}")

    plt.xlabel("t")
    plt.ylabel("W(t)")
    plt.title("Wiener process: direct simulation vs. KL approximation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("wiener_process_comparison.png", dpi=150)
    plt.show()

    # ✓ TODO: visualize first eigenvalues and eigenfunctions.
    fig = plot_eigenpairs(WP, n_terms=1000, t_grid=t_grid)
    # plt.savefig("wiener_eigenpairs_1000.png", dpi=150)
    fig.show()

    fig = plot_eigenpairs(WP, n_terms=5, t_grid=t_grid)
    ax_eigenfunctions = fig.axes[1]  # the eigenfunctions subplot
    for m, line in enumerate(ax_eigenfunctions.get_lines(), start=1):
        line.set_label(fr"$\phi_{{{m}}}(t)$")
    ax_eigenfunctions.legend()
    # plt.savefig("wiener_eigenpairs_5.png", dpi=150)
    fig.show()

