import matplotlib.pyplot as plt
import numpy as np
import itertools
from math import ceil


def label_xy():
    plt.xlabel("X")
    plt.ylabel("Y")


def plot_solution(sol, x, y):
    plt.clf()
    label_xy()
    fig, axs = plt.subplots(1, sol.shape[-1])
    for i, ax in enumerate(axs):
        ax.contourf(
            x[:, :, i], y[:, :, i], sol[:, :, i], levels=15, cmap="nipy_spectral"
        )
        # plt.clabel(contour, colors="black", inline=True, fontsize="8")
        ax.scatter(x, y, color="white", s=0.1)
        # ax.colorbar(contour)
    # fig.title("Temperature of Plate [C]")
    return fig


def plot_mesh(x, y, z, connectivity_matrix):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, y, z)
    # x = x.reshape((-1,))
    # y = y.reshape((-1,))
    # z = z.reshape((-1,))
    # colors = itertools.cycle(["r","b","k","g",])
    # for row in range(connectivity_matrix.shape[0]):
    #     color = next(colors)
    #     idx = connectivity_matrix[row, :]
    #     xe, ye, ze = list(x[idx]), list(y[idx]), list(z[idx])
    #     xs, xe = xe[:len(xe)//2], xe[len(xe)//2:]
    #     ys, ye = ye[:len(ye)//2], ye[len(ye)//2:]
    #     zs, ze = ze[:len(ze)//2], ze[len(ze)//2:]
    #     xe.append(xe[0])
    #     ye.append(ye[0])
    #     ze.append(ze[0])
    #     xs.append(xs[0])
    #     ys.append(ys[0])
    #     zs.append(zs[0])
    #
    #     ax.plot(xs, ys, zs, color=color)
    #     ax.plot(xe, ye, ze, color=color)

    return fig


def plot_bc(x, y, z, bc_list):
    plt.clf()
    label_xy()
    side_size = ceil(bc_list.shape[1] ** (1 / 3))
    bc_list = bc_list.reshape((side_size, side_size, side_size))

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    pos = ax.scatter(x, y, z, c=bc_list, cmap="nipy_spectral")
    fig.colorbar(label="values", mappable=pos, ax=ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return plt
