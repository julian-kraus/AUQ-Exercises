import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.sampling import compute_rmse


def sample_normal(
    n_samples: int, mu_target: npt.NDArray, V_target: npt.NDArray, seed: int = 42
) -> npt.NDArray:
    # ✓ TODO: generate samples from multivariate normal distribution.
    # ====================================================================
    distr = cp.MvNormal(mu_target, V_target)
    samples = distr.sample(n_samples, rule="random", seed=seed)
    # ====================================================================
    return samples


def compute_moments(samples: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    # ✓ TODO: estimate mean and covariance of the samples.
    # ====================================================================
    mean = np.mean(samples, axis=1)
    covariance = np.cov(samples, ddof=1)
    # ====================================================================
    return mean, covariance


if __name__ == "__main__":
    # ✓ TODO: define the parameters of the simulation.
    # ====================================================================
    means = np.array([-0.4, 1.1])
    covs = np.array([[2, 0.4], [0.4, 1]])
    sample_sizes = [10, 100, 1000, 10000]
    seeds = [41, 42, 43]

    for seed in seeds:
        mean_errors = []
        cov11_errors = []
        cov12_errors = []
        rsmes = []

        samples = {
            n: sample_normal(n, means, covs, seed=seed)
            for n in sample_sizes
        }
        # ====================================================================

        # ✓ TODO: estimate mean, covariance, and compute the required errors.
        # ====================================================================
        for n in sample_sizes:
            values = samples[n]
            mean, cov = compute_moments(values)
            rmse = compute_rmse(values)

            # Mean error (first value)
            mean_error = abs(mean[0] - means[0])

            # Covariance errors (first diagonal and upper off-diagonal value)
            cov11_error = abs(cov[0, 0] - covs[0, 0])
            cov12_error = abs(cov[0, 1] - covs[0, 1])

            mean_errors.append(mean_error)
            cov11_errors.append(cov11_error)
            cov12_errors.append(cov12_error)
            rsmes.append(rmse)
            print(f"----------Size {n}----------")
            print(f"Mean: {mean}")
            print(f"Cov: {cov}")
            print(f"RMSE: {rmse}")

        # ====================================================================

        # ✓ TODO: plot the results on the log-log scale.
        # ====================================================================
        plt.figure()

        plt.loglog(sample_sizes, mean_errors, marker="o", label="Mean error")
        plt.loglog(sample_sizes, cov11_errors, marker="o", label="Cov(1,1) error")
        plt.loglog(sample_sizes, cov12_errors, marker="o", label="Cov(1,2) error")

        plt.title(f"Estimator Absolute Errors (seed={seed})")
        plt.xlabel("Sample size")
        plt.ylabel("Absolute error")
        plt.legend()
        plt.grid(True)

        plt.savefig(f"plots/assignment_1_2_ae_seed{seed}.png")

        rmse1 = [r[0] for r in rsmes]
        rmse2 = [r[1] for r in rsmes]

        plt.figure()

        plt.loglog(sample_sizes, rmse1, marker="o", label="RMSE dim 1")
        plt.loglog(sample_sizes, rmse2, marker="o", label="RMSE dim 2")

        plt.title(f"RMSE of Mean Estimator (seed={seed})")
        plt.xlabel("Sample size")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True)

        plt.show()
        # plt.savefig(f"plots/assignment_1_2_rmse_seed{seed}.png")
    # ====================================================================
