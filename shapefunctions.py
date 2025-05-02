from typing import Callable
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


def triangle(i):
    """Shape functions for a 3-node triangle."""
    r, s = i
    NN = np.array([(1 / 2) * (-r - s), (1 / 2) * (1 + r), (1 / 2) * (1 + s)])

    # Derivatives
    Nr = np.array([-1 / 2, 1 / 2, 0])
    Ns = np.array([-1 / 2, 0, 1 / 2])

    return NN, Nr, Ns


def quad(i):
    """Shape functions for a 4-node quadrilateral."""
    r, s = i
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


def hexahedron(i: tuple[float, float, float]):
    ξ, η, ζ = i

    # Shape functions N_1 to N_8
    NN = np.array(
        [
            1 / 8 * (1 - ξ) * (1 - η) * (1 - ζ),
            1 / 8 * (1 + ξ) * (1 - η) * (1 - ζ),
            1 / 8 * (1 + ξ) * (1 + η) * (1 - ζ),
            1 / 8 * (1 - ξ) * (1 + η) * (1 - ζ),
            1 / 8 * (1 - ξ) * (1 - η) * (1 + ζ),
            1 / 8 * (1 + ξ) * (1 - η) * (1 + ζ),
            1 / 8 * (1 + ξ) * (1 + η) * (1 + ζ),
            1 / 8 * (1 - ξ) * (1 + η) * (1 + ζ),
        ]
    )

    Nξ = np.array(
        [
            -1 / 8 * (1 - η) * (1 - ζ),
            1 / 8 * (1 - η) * (1 - ζ),
            1 / 8 * (1 + η) * (1 - ζ),
            -1 / 8 * (1 + η) * (1 - ζ),
            -1 / 8 * (1 - η) * (1 + ζ),
            1 / 8 * (1 - η) * (1 + ζ),
            1 / 8 * (1 + η) * (1 + ζ),
            -1 / 8 * (1 + η) * (1 + ζ),
        ]
    )

    Nη = np.array(
        [
            -1 / 8 * (1 - ξ) * (1 - ζ),
            -1 / 8 * (1 + ξ) * (1 - ζ),
            1 / 8 * (1 + ξ) * (1 - ζ),
            1 / 8 * (1 - ξ) * (1 - ζ),
            -1 / 8 * (1 - ξ) * (1 + ζ),
            -1 / 8 * (1 + ξ) * (1 + ζ),
            1 / 8 * (1 + ξ) * (1 + ζ),
            1 / 8 * (1 - ξ) * (1 + ζ),
        ]
    )

    Nζ = np.array(
        [
            -1 / 8 * (1 - ξ) * (1 - η),
            -1 / 8 * (1 + ξ) * (1 - η),
            -1 / 8 * (1 + ξ) * (1 + η),
            -1 / 8 * (1 - ξ) * (1 + η),
            1 / 8 * (1 - ξ) * (1 - η),
            1 / 8 * (1 + ξ) * (1 - η),
            1 / 8 * (1 + ξ) * (1 + η),
            1 / 8 * (1 - ξ) * (1 + η),
        ]
    )

    return NN, Nξ, Nη, Nζ


ELEMENT_TYPES = {
    "line": line,
    "triangle": triangle,
    "quad": quad,
    "curve": curve,
    "hexahedron": hexahedron,
}


def get_shape_func(el_type: str) -> Callable:
    """Returns the appropriate shape function based on element type."""
    return ELEMENT_TYPES[el_type]
