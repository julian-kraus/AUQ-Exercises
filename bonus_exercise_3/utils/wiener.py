from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class WienerProcess:
    mu: float
    T: float | None = None
    n_points: float | None = None
    t_grid: npt.NDArray | None = None

    def __post_init__(self):
        if self.T is None and self.n_points is None:
            self.T = self.t_grid[-1]
            self.n_points = len(self.t_grid)
        if self.t_grid is None:
            self.t_grid = np.linspace(0, self.T, self.n_points)

    def generate(self, n_samples: int, rng: np.random.Generator):

        # ✓ TODO: generate n_samples realizations of the Wiener process
        # Compute dt
        dt = np.diff(self.t_grid)

        increments = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_samples, self.n_points - 1))

        process = np.zeros((n_samples, self.n_points))
        # Compute each row as a cumultative sum from the previous row(s)
        process[:, 1:] = np.cumsum(increments, axis=1)
        # Shift by mean
        process += self.mu

        return process

    def approximate_kl(self, n_samples: int, M: int, rng: np.random.Generator):

        # ✓ TODO: generate n_samples realizations of the Wiener process
        # using the Karhunen-Loève expansion with M terms.
        # Generate mode indices, start at m, end at M incl.
        m = np.arange(1, M + 1)
        # Generate random var zeta for each time step and each mode
        zeta = rng.normal(0, 1, size=(n_samples, M))

        term = (m[:, None] - 0.5) * np.pi
        phi_eig = np.sqrt(2 * self.T) * np.sin(term / self.T * self.t_grid[None, :]) / term

        processes = zeta @ phi_eig
        # Shift by mean
        processes += self.mu
        return processes


        processes = zeta @ phi_eig
        # Shift by mean
        processes += self.mu
        return processes

    def kl_eigenvalues(self, M: int):

        # ✓ TODO: compute the first M eigenvalues of the Wiener process.
        # Generate mode indices, start at m, end at M incl.
        m = np.arange(1, M + 1)
        return 1.0 / ((m - 0.5) ** 2 * np.pi ** 2)

    def kl_eigenfunctions(self, M: int):

        # ✓ TODO: compute the first M eigenfunctions of the Wiener process.
        # It might be more conveniet to return a callable function that
        # returns evaluations of the first M eigenfunctions for the provided
        # time points.
        # Generate mode indices, start at m, end at M incl.
        m = np.arange(1, M + 1)
        term = (m[None, :] - 0.5) * np.pi
        # Return a callable for a certain time grid
        return lambda t_grid: np.sqrt(2) * np.sin(term * t_grid[:, None])

    def kl_eigenpairs(self, M: int):
        eigenvalues = self.kl_eigenvalues(M)
        eigenfunctions = self.kl_eigenfunctions(M)
        return eigenvalues, eigenfunctions