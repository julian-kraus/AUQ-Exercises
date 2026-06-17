import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def calculate_polynoms_product(distr: cp.Distribution, n: int) -> npt.NDArray:
    # TODO-DONE: generate the first n orthonormal polynomials w.r.t. the given
    # distribution and compute the expected value of their inner products.
    # ====================================================================
    expected_prod = np.zeros((n + 1, n + 1)) #placeholder

    #Generate an orthonormal expansion up to n-th order
    orthonormal_expansion = cp.generate_expansion(n, distr, normed=True)

    for i in range(n + 1):
        for j in range(n + 1):
            expected_prod [i][j] = cp.E(orthonormal_expansion[i]*orthonormal_expansion[j],distr)
    # ====================================================================
    return expected_prod


if __name__ == "__main__":
    # TODO-DONE: define the parameters of the simulation.
    # ====================================================================
    # pass # placeholder

    distr_1 = cp.Uniform(-1, 1)
    distr_2 = cp.Normal(5, 1)
    n = 10
    I = np.identity(n+1)
    # ====================================================================

    # Compute the inner products.
    # ====================================================================
    # pass # placeholder

    expected_prod_1 = calculate_polynoms_product(distr_1, n)
    expected_prod_2 = calculate_polynoms_product(distr_2, n)
    # ====================================================================

    # Visualize the results.
    # ====================================================================
    # pass # placeholder

    diff_identity_1 = expected_prod_1 - I
    diff_identity_2 = expected_prod_2 - I

    #Using imshow to display the differences as images
    #Expecting the data to be 0 or deviate around 0, so choosing diverging colormaps
    #Also, giving a range for normalisation of our difference scalars to the colormaps
    fig1, axes1 = plt.subplots()
    axes1.set_title("Difference b/w <phi_i(x),phi_j(x)>_uni and I")
    #Very small differences here, so using 1e-10 as the scale
    imshow1 = axes1.imshow(diff_identity_1,cmap="PiYG",aspect='auto',vmin=-1e-10, vmax=1e-10)
    fig1.colorbar(imshow1, ax=axes1)
    fig1.savefig("a2_diff_colormap_uniform.png")

    fig2, axes2 = plt.subplots()
    axes2.set_title("Difference b/w <phi_i(x),phi_j(x)>_nor and I")
    #Not as small differences here, so using 1e-5 as the scale
    imshow2 = axes2.imshow(diff_identity_2,cmap="PiYG",aspect='auto',vmin=-1e-5, vmax=1e-5)
    fig2.colorbar(imshow2, ax=axes2)
    fig2.savefig("a2_diff_colormap_normal.png")
    # ====================================================================
