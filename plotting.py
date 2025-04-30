import matplotlib.pyplot as plt
import numpy as np


def label_xy():
    plt.xlabel("X")
    plt.ylabel("Y")


def plot_solution(sol, x, y):
    plt.clf()
    label_xy()
    contour = plt.contourf(x, y, sol, levels=15, cmap="nipy_spectral")
    # plt.clabel(contour, colors="black", inline=True, fontsize="8")
    plt.scatter(x, y, color="white", s=0.1)
    plt.colorbar(contour)
    plt.title("Temperature of Plate [C]")
    return plt


def plot_mesh(x, y, connectivity_matrix):
    plt.clf()
    plt.scatter(x, y)
    label_xy()
    x = x.reshape((-1,))
    y = y.reshape((-1,))
    for row in range(connectivity_matrix.shape[0]):
        idx = connectivity_matrix[row, :]
        xe, ye = list(x[idx]), list(y[idx])
        xe.append(xe[0])
        ye.append(ye[0])
        plt.plot(xe, ye)

    return plt


def plot_bc(x, y, bc_list):
    plt.clf()
    label_xy()
    side_size = int(np.sqrt(bc_list.shape[1]))
    bc_list = bc_list.reshape((side_size, side_size))
    contour = plt.contourf(x, y, bc_list, cmap="nipy_spectral")
    plt.colorbar(contour)
    return plt
