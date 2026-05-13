from collections import defaultdict
import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.oscillator import Oscillator


def load_reference(filename: str) -> tuple[float, float]:
    # TODO-DONE: load reference values for the mean and variance.
    # ====================================================================
    # mean, var = 0, 1 #placeholder

    # open the file with the two references in read mode, use "with" for auto closing of file
    with open(filename, "r") as file_with_refs:
        # read the two lines, convert from string to float and the values in the (mean,variance) tuple
        mean, var = float(file_with_refs.readline()), float(file_with_refs.readline())
    # ====================================================================
    return mean, var


def simulate(
    t_grid: npt.NDArray,
    omega_distr: cp.Distribution,
    n_samples: int,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
    rule="random",
    seed=42,
) -> npt.NDArray:
    # TODO-DONE: simulate the oscillator with the given parameters and return
    # generated solutions.
    # ====================================================================
    # sample_solutions = np.zeros((n_samples, len(t_grid))) #placeholder

    # The deterministic oscillator parameters are:
    c = model_kwargs["c"]
    k = model_kwargs["k"]
    f = model_kwargs["f"]
    y0 = init_cond["y0"]
    y1 = init_cond["y1"]
    method = "odeint" # it's faster than Euler for larger sample sizes as seen in the tutorial

    # The stochastic oscillator parameter is omega
    # It should be modeled as a random vector
    # We are given the distribution of omega
    # Taking N samples for omega
    n_omegas = omega_distr.sample(size=np.array([n_samples]), rule=rule, seed=seed)

    # Return data structure: a 2D matrix
    # Each of the N rows corresponds to a simulated oscillator for a sample omega_i
    # Each row contains an evaluated value for each of the time points in T_grid
    sample_solutions = np.empty((n_samples, len(t_grid)))

    # Simulating (evaluating) the oscillator N times
    for _i in range(n_samples):
        oscillator = Oscillator(c=c, k=k, f=f, omega=n_omegas[_i])
        sample_solutions[_i] = oscillator.discretize(method=method, y0=y0, y1=y1 , t_grid=t_grid)
    # ====================================================================
    return sample_solutions


def compute_errors(
    samples: npt.NDArray, mean_ref: float, var_ref: float
) -> tuple[float, float]:
    # TODO-DONE: compute the relative errors of the mean and variance
    # estimates.
    # ====================================================================
    # mean_error, var_error = 0, 0 #placeholder

    # Using numpy, some quick statistics
    mean_of_estimation = np.mean(samples, dtype=float)
    # Taking the unbiased estimator as in the tutor exercises (normalization with 1/n-1)
    var_of_estimation = np.var(samples, ddof=1, dtype=float)

    # Using the relative error formula from the task sheet
    mean_error = np.abs(1-mean_of_estimation/mean_ref)
    var_error = np.abs(1-var_of_estimation/var_ref)

    # ====================================================================
    return mean_error, var_error


if __name__ == "__main__":
    # TODO-DONE: define the parameters of the simulations.
    # ====================================================================
    #pass #placeholder

    # Oscillator parameters
    # The deterministic oscillator parameters are:
    # c = 0.5, k = 2.0, f = 0.5
    model_kwargs = defaultdict(float)
    model_kwargs["c"] = 0.5
    model_kwargs["k"] = 2.0
    model_kwargs["f"] = 0.5
    # y0 = 0.5, y1 = 0.0
    init_cond = defaultdict(float)
    init_cond["y0"] = 0.5
    init_cond["y1"] = 0.0
    # The stochastic oscillator parameter is omega ~ U(0.95, 1.05)
    omega_distr = cp.Uniform(0.95, 1.05)

    # Time points from 0 to 10 with t_delta = 0.01
    # Arrange creates values in half-open interval [start, stop),
    # hence padding the stop with an additional step
    t_grid = np.arange(0, 10.01, 0.01)

    # Number of simulations:
    Ns = [10, 100, 1000, 10000]

    # Time step for convergence analysis: y(10) (penultimate subtask)
    t_analyzed = 10

    # Plot 10 solution trajectories for N=10 for the ODE solution with standard MC (last subtask)
    solutions_traj_plot = 10
    n_for_plot_traj = 10
    # ====================================================================

    # TODO-DONE: run the simulations.
    # ====================================================================
    #pass #placeholder

    # Monte Carlo
    simulations_MonteCarlo = defaultdict(lambda: np.ndarray(0))
    for n in Ns:
        simulations_MonteCarlo[n] = simulate(
            t_grid=t_grid,
            omega_distr=omega_distr,
            n_samples=n,
            model_kwargs=model_kwargs,
            init_cond=init_cond
        )

    # Quasi-Monte-Carlo based on Halton sequences
    simulations_QuasiMonteCarlo_Halton = defaultdict(lambda: np.ndarray(0))
    for n in Ns:
        simulations_QuasiMonteCarlo_Halton[n] = simulate(
            t_grid=t_grid,
            omega_distr=omega_distr,
            n_samples=n,
            model_kwargs=model_kwargs,
            init_cond=init_cond,
            rule="H" # pass the Halton rule for the sampling of omega
        )
    # ====================================================================

    # TODO-DONE: compute the statistics.
    # ====================================================================
    #pass #placeholder

    # Task is: To analyze the convergence,
    # compute the mean and the variance of y(t_analyzed)
    # and compare the results to the reference values

    # In the specific Assignment 4, t = 10 is the last time point for the simulation with index 1000
    # But it could be that we want to see statistics for another time step
    # Calculate the index of t_analyzed in the time point grid
    t_analyzed_index = np.where(np.isclose(t_grid,t_analyzed))[0][0]

    # Extract the columns for y at t_analyzed
    simulations_MonteCarlo_at_t_analyzed = defaultdict(lambda: np.ndarray(0))
    simulations_QuasiMonteCarlo_Halton_at_t_analyzed = defaultdict(lambda: np.ndarray(0))
    for n in Ns:
        simulations_MonteCarlo_at_t_analyzed[n] = simulations_MonteCarlo[n][:,t_analyzed_index]
        simulations_QuasiMonteCarlo_Halton_at_t_analyzed[n] = simulations_QuasiMonteCarlo_Halton[n][:,t_analyzed_index]

    # Get the reference values with the helper method
    mean_ref, var_ref = load_reference("data/oscillator_ref.txt")

    # Statistics for Monte Carlo - mean and variance error for each N and for y(t_analyzed)
    # Save the results in an array instead of map because the size is clear and also ease of plotting later
    mean_errors_MonteCarlo = np.empty(len(Ns))
    var_errors_MonteCarlo = np.empty(len(Ns))
    _index_MC = 0
    for n in Ns:
        error_tuple = compute_errors(
            samples=simulations_MonteCarlo_at_t_analyzed[n],
            mean_ref=mean_ref,
            var_ref=var_ref
        )
        mean_errors_MonteCarlo[_index_MC] = error_tuple[0]
        var_errors_MonteCarlo[_index_MC] = error_tuple[1]
        _index_MC = _index_MC + 1

    # Statistics for Quasi-Monte-Carlo based on Halton - mean and variance error for each N and for y(t_analyzed)
    # Save the results in an array instead of map because the size is clear and also ease of plotting later
    mean_errors_QuasiMonteCarlo_Halton = np.empty(len(Ns))
    var_errors_QuasiMonteCarlo_Halton = np.empty(len(Ns))
    _index_QMC = 0
    for n in Ns:
        error_tuple = compute_errors(
            samples=simulations_QuasiMonteCarlo_Halton_at_t_analyzed[n],
            mean_ref=mean_ref,
            var_ref=var_ref
        )
        mean_errors_QuasiMonteCarlo_Halton[_index_QMC] = error_tuple[0]
        var_errors_QuasiMonteCarlo_Halton[_index_QMC] = error_tuple[1]
        _index_QMC = _index_QMC + 1
    # ====================================================================

    # TODO-DONE: plot the results on the log-log scale.
    # ====================================================================
    #pass #placeholder

    #Plotting relative error of the mean over number of samples
    fig1, plot1 = plt.subplots()
    plot1.set_title("Mean Relative Err")
    plot1.set_xlabel("Number of samples")
    plot1.set_ylabel("Mean Rel Error")
    plot1.loglog(Ns, mean_errors_MonteCarlo, label="MonteCarlo")
    plot1.loglog(Ns, mean_errors_QuasiMonteCarlo_Halton, label="QuasiMonteCarlo_Halton")
    plot1.legend()
    fig1.tight_layout()
    fig1.savefig("a4_mean_rel_err_plot.png")

    #Plotting relative error of the variance over number of samples
    fig2, axes2 = plt.subplots()
    axes2.set_title("Variance Relative Err")
    axes2.set_xlabel("Number of samples")
    axes2.set_ylabel("Var Rel Error")
    axes2.loglog(Ns, var_errors_MonteCarlo, label="MonteCarlo")
    axes2.loglog(Ns, var_errors_QuasiMonteCarlo_Halton, label="QuasiMonteCarlo_Halton")
    axes2.legend()
    fig2.tight_layout()
    fig2.savefig("a4_mean_var_err_plot.png")
    # ====================================================================

    # TODO-DONE: plot sampled trajectories.
    # ====================================================================
    #pass #placeholder

    # Task is:  Report a plot where you show 10 solution trajectories,
    # i.e., 10 solutions of the ODE plotted over the time points t ∈ [0, 10]
    # which result from N=10 different samples of omega with the standard Monte Carlo method.

    # In the task, we want to show 10 solutions for N=10
    # But we may want to generalize this of other n_to_plot_traj or n_solutions_traj_plot
    # Plot the results
    fig, axes = plt.subplots()
    axes.set_title("ODE solutions from different samples")
    axes.set_xlabel("Time")
    axes.set_ylabel("ODE solution")
    for i in range(solutions_traj_plot):
        axes.plot(t_grid, simulations_MonteCarlo[n_for_plot_traj][i],  linestyle="--", label = i)
    axes.legend(title="Sample")
    fig.tight_layout()
    fig.savefig("a4_ODE_solutions.png")
    # ====================================================================