import matplotlib.pyplot as plt
import numpy as np
import itertools
from math import ceil

from preprocess import Mesh


def label_xy():
    plt.xlabel("X")
    plt.ylabel("Y")


def plot_solution(sol, x, y):
    """
    Plots the solution with contour plots for each slice,
    ensuring all plots share the same color scale.

    Args:
        sol (np.ndarray): The solution array with shape (nx, ny, num_slices).
        x (np.ndarray): The x-coordinates array with shape (nx, ny, num_slices).
        y (np.ndarray): The y-coordinates array with shape (nx, ny, num_slices).

    Returns:
        matplotlib.figure.Figure: The figure object containing the plots.
    """
    plt.clf()
    global_min = np.min(sol)
    global_max = np.max(sol)

    fig, axs = plt.subplots(1, sol.shape[-1], figsize=(12, 4))

    if sol.shape[-1] == 1:
        axs = [axs]

    num_levels = 20

    # Generate levels based on the global min and max
    levels = np.linspace(global_min, global_max, num_levels)

    for i, ax in enumerate(axs):
        # Create the contour plot using the global levels and color limits
        pos = ax.contourf(
            x[:, :, i], y[:, :, i], sol[:, :, i],
            levels=levels, # Use globally defined levels
            cmap="nipy_spectral",
            vmin=global_min, # Set minimum color value
            vmax=global_max  # Set maximum color value
        )

        # Optional: Add scatter points for grid points
        ax.scatter(x[:, :, i], y[:, :, i], color="white", s=0.1)

        if i == sol.shape[-1] - 1:
             fig.colorbar(pos, ax=axs, label="Temperature")
        if i == 0:
            ax.set_ylabel('Y-axis')
        else:
            ax.set_yticklabels([]) 


        # Set titles for each subplot if needed
        ax.set_title(f'Slice {i+1}')
        ax.set_xlabel('X-axis')
        ax.set_aspect('equal', adjustable='box')

    return fig


def plot_mesh(x, y, z, connectivity_matrix):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, y, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # el_conn = connectivity_matrix["quad"]
    # for row in range(el_conn.shape[0]):
    #     idx = el_conn[row, :]
    #     xe, ye, ze = list(x[idx]), list(y[idx]), list(z[idx])
    #     xe.append(xe[0])
    #     ye.append(ye[0])
    #     ze.append(ze[0])
    #     ax.plot(xe, ye, ze, color="r")

    x = x.reshape((-1,))
    y = y.reshape((-1,))
    z = z.reshape((-1,))
    colors = itertools.cycle(
        [
            "r",
            "b",
            "k",
            "g",
            "cyan",
        ]
    )
    el_conn = connectivity_matrix["hexahedron"]
    for row in range(el_conn.shape[0]):
        color = next(colors)
        idx = el_conn[row, :]
        xe, ye, ze = list(x[idx]), list(y[idx]), list(z[idx])
        xs, xe = xe[: len(xe) // 2], xe[len(xe) // 2 :]
        ys, ye = ye[: len(ye) // 2], ye[len(ye) // 2 :]
        zs, ze = ze[: len(ze) // 2], ze[len(ze) // 2 :]
        xe.append(xe[0])
        ye.append(ye[0])
        ze.append(ze[0])
        xs.append(xs[0])
        ys.append(ys[0])
        zs.append(zs[0])

        ax.plot(xs, ys, zs, color=color)
        ax.plot(xe, ye, ze, color=color)

        xes = [
            [x[idx][0], x[idx][0]],
            [x[idx][2], x[idx][2]],
            [x[idx][0], x[idx][0]],
            [x[idx][2], x[idx][2]],
        ]
        yes = [
            [y[idx][0], y[idx][4]],
            [y[idx][0], y[idx][4]],
            [y[idx][0], y[idx][4]],
            [y[idx][0], y[idx][4]],
        ]
        zes = [
            [z[idx][0], z[idx][0]],
            [z[idx][0], z[idx][0]],
            [z[idx][1], z[idx][1]],
            [z[idx][1], z[idx][1]],
        ]
        for xi, yi, zi in zip(xes, yes, zes):
            ax.plot(xi, yi, zi, color=color)

    return fig


def plot_bc(mesh: Mesh):
    plt.clf()
    label_xy()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    idx = np.where(mesh.BC_fix_list == 1)[-1]
    pos = ax.scatter(
        mesh.x[idx],
        mesh.y[idx],
        mesh.z[idx],
        c=mesh.BC_g_list[0, idx],
        cmap="nipy_spectral",
    )

    el_conn = mesh.element_connectivity["quad"]
    for row in range(el_conn.shape[0]):
        idx = el_conn[row, :]
        xe, ye, ze = list(mesh.x[idx]), list(mesh.y[idx]), list(mesh.z[idx])
        xe.append(xe[0])
        ye.append(ye[0])
        ze.append(ze[0])
        ax.plot(xe, ye, ze, color="r")

    fig.colorbar(label="values", mappable=pos, ax=ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return plt
