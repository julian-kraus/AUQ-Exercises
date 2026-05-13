import numpy as np
import numpy.typing as npt


def load_grades(filename: str) -> npt.NDArray:
    # ✓ TODO: read grades from the file.
    # ====================================================================
    grades = np.loadtxt(filename, dtype=float)
    # ====================================================================
    return grades


def python_compute(array: npt.NDArray) -> tuple[float, float]:
    # ✓ TODO: compute the mean and the variance using standard Python.
    # ====================================================================
    mean, var = 0, 1
    mean = sum(array) / len(array)
    if len(array) > 1:
        var = sum((x - mean) ** 2 for x in array) / (len(array) - 1)
    # ====================================================================
    return mean, var


def numpy_compute(array: npt.NDArray, ddof: int = 0) -> tuple[float, float]:
    # ✓ TODO: compute the mean and the variance using numpy.
    # ====================================================================
    mean, var = 0, 1
    mean = np.mean(array, dtype=float)
    var = np.var(array, ddof=ddof, dtype=float)
    # ====================================================================
    return mean, var


if __name__ == "__main__":
    # ✓ TODO: load the grades from the file, compute the mean and the
    # variance using both implementations and report the results.
    # ====================================================================
    grades = load_grades("data/G.txt")
    mean_python, var_python = python_compute(grades)
    mean_numpy_0, var_numpy_0 = numpy_compute(grades, ddof=0)
    mean_numpy_1, var_numpy_1 = numpy_compute(grades, ddof=1)
    print("Python--------------")
    print(f"Mean: {mean_python}")
    print(f"Variance: {var_python}")
    print("Numpy with ddof=0--------------")
    print(f"Mean: {mean_numpy_0}")
    print(f"Variance: {var_numpy_0}")
    print("Numpy with ddof=1--------------")
    print(f"Mean: {mean_numpy_1}")
    print(f"Variance: {var_numpy_1}")
    # ====================================================================
