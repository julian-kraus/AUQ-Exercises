from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def exp_cov_fn(x: npt.NDArray, y: npt.NDArray, scale: float) -> npt.NDArray:
    """Computes the exponential covariance function between two sets of points."""

    # TODO: compute the exponential covariance function.
    distances = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    return np.exp(-distances / scale)


def squared_exp_cov_fn(x: npt.NDArray, y: npt.NDArray, scale: npt.NDArray):
    """Computes the squared exponential covariance function between two sets of points."""

    # TODO: compute the squared exponential covariance function.
    squared_distances = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return np.exp(-squared_distances / (2.0 * scale**2))

def get_xy_mesh(
    x_lims: tuple[float, float],
    y_lims: tuple[float, float],
    x_mesh_size: int,
    y_mesh_size: int,
) -> npt.NDArray:
    """Creates a 2D mesh grid for the given limits and mesh sizes."""
    x_step = (x_lims[1] - x_lims[0]) / x_mesh_size
    y_step = (y_lims[1] - y_lims[0]) / y_mesh_size
    x_grid = np.arange(x_lims[0] + x_step / 2, x_lims[1], x_step)
    y_grid = np.arange(y_lims[0] + y_step / 2, y_lims[1], y_step)
    mesh = np.stack(np.meshgrid(x_grid, y_grid), axis=-1)
    return mesh


def sample(mesh, mean_fn, cov_fn, n_samples, rng, reg_scale=1e-7):
    """Samples from a Gaussian process defined by the mean and covariance functions."""
    points = mesh.reshape(-1, mesh.shape[-1])
    mean = mean_fn(points)
    covariance = cov_fn(points, points)

    jitter = 0.0
    eye = np.eye(points.shape[0])
    for attempt in range(8):
        try:
            L = np.linalg.cholesky(covariance + jitter * eye)
            break
        except np.linalg.LinAlgError:
            jitter = reg_scale if attempt == 0 else 10.0 * jitter
    else:
        raise np.linalg.LinAlgError("Cholesky decomposition failed after jittering.")

    psi = rng.standard_normal((points.shape[0], n_samples))
    fields = mean[:, None] + L @ psi
    # TODO: sample a Gaussian field suing the Cholesky decomposition.
    return fields.T.reshape(n_samples, *mesh.shape[:-1])

def plot_samples(samples, x_lims, y_lims, title, vmin, vmax):
    """Plots the samples from the Gaussian process."""
    n_plots = len(samples)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots + 0.8, 5))
    if n_plots == 1:
        axes = [axes]

    image = None
    for i, (ax, sample) in enumerate(zip(axes, samples), start=1):
        image = ax.imshow(
            sample,
            cmap="coolwarm",
            origin="lower",
            extent=(*x_lims, *y_lims),
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Sample {i}")

    fig.suptitle(title)
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.1, top=0.82, wspace=0.25)
    colorbar_axis = fig.add_axes([0.92, 0.16, 0.015, 0.66])
    fig.colorbar(image, cax=colorbar_axis)
    return fig


if __name__ == "__main__":
    # TODO: set the condiguration.
    x_lims, y_lims = (0.0, 1.0), (0.0, 1.0)
    x_mesh_size, y_mesh_size = 40, 40
    scale = 0.2
    mean = 0.1
    seed = 42
    n_samples = 3
    rng = np.random.default_rng(seed)

    # TODO: create a 2D mesh.

    # TODO: sample from the Gaussian process with different kernels.

    # TODO: plot the samples.

    mesh = get_xy_mesh(x_lims, y_lims, x_mesh_size, y_mesh_size)
    mean_fn = lambda points: np.full(points.shape[0], mean)

    exp_cov = partial(exp_cov_fn, scale=scale)
    squared_exp_cov = partial(squared_exp_cov_fn, scale=scale)

    exp_samples = sample(mesh, mean_fn, exp_cov, n_samples, rng)
    squared_exp_samples = sample(mesh, mean_fn, squared_exp_cov, n_samples, rng)

    vmin = min(exp_samples.min(), squared_exp_samples.min())
    vmax = max(exp_samples.max(), squared_exp_samples.max())
    span = max(abs(vmin - mean), abs(vmax - mean))
    vmin, vmax = mean - span, mean + span

    fig = plot_samples(
        exp_samples,
        x_lims,
        y_lims,
        "Exponential covariance",
        vmin,
        vmax,
    )
    fig.savefig(
        "outputs/assignment_2_exponential_samples.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()

    fig = plot_samples(
        squared_exp_samples,
        x_lims,
        y_lims,
        "Squared exponential covariance",
        vmin,
        vmax,
    )
    fig.savefig(
        "outputs/assignment_2_squared_exponential_samples.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
