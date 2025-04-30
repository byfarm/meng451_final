import numpy as np


def line(r):
    """Shape functions for a 2-node line."""
    if isinstance(r, (list, tuple, np.ndarray)):
        r = r[0]

    NN = np.array([(1 - r) / 2, (1 + r) / 2])
    Nr = np.array([-1 / 2, 1 / 2])
    return NN, Nr


def curve(ξ):
    """
    Shape function for 3 node quadratic
    Make sure x=0 is the middle node "hours later"
    """
    NN = np.array([ξ * (ξ - 1) / 2, 1 - ξ**2, ξ * (ξ + 1) / 2])
    Nξ = np.array([ξ - 1 / 2, -2 * ξ, ξ + 1 / 2])
    return NN, Nξ


def triangle(r, s):
    """Shape functions for a 3-node triangle."""
    NN = np.array([(1 / 2) * (-r - s), (1 / 2) * (1 + r), (1 / 2) * (1 + s)])

    # Derivatives
    Nr = np.array([-1 / 2, 1 / 2, 0])
    Ns = np.array([-1 / 2, 0, 1 / 2])

    return NN, Nr, Ns


def quad(r, s):
    """Shape functions for a 4-node quadrilateral."""
    NN = np.array(
        [
            (1 / 4) * (-1 + r) * (-1 + s),
            (-1 / 4) * (1 + r) * (-1 + s),
            (1 / 4) * (1 + r) * (1 + s),
            (-1 / 4) * (-1 + r) * (1 + s),
        ]
    )

    # Derivatives
    Nr = np.array(
        [(1 / 4) * (-1 + s), (1 / 4) * (1 - s), (1 / 4) * (1 + s), (1 / 4) * (-1 - s)]
    )

    Ns = np.array(
        [(1 / 4) * (-1 + r), (1 / 4) * (-1 - r), (1 / 4) * (1 + r), (1 / 4) * (1 - r)]
    )

    return NN, Nr, Ns


def shapefunc(el_type):
    """Returns the appropriate shape function based on element type."""
    match el_type:
        case "line":
            return line
        case "triangle":
            return triangle
        case "quad":
            return quad
        case "curve":
            return curve
        case _:
            raise ValueError("Element type not found")
