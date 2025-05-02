import numpy as np
from collections import namedtuple
from preprocess import build_mesh
import quadrature as quad
from cube import assemble_stiffness, assemble_rhs
from plotting import plot_solution, plot_mesh, plot_bc
import sys

np.set_printoptions(
    threshold=sys.maxsize, linewidth=1000000000, precision=2, suppress=True
)

# Define properties
Properties = namedtuple("Properties", ["w", "h", "d", "K"])
prop = Properties(w=1, h=1, d=1, K=1)


def build():
    # Define quad rule
    quad_rules = {
        "hexahedron": quad.gauss_legendre_1d(2),
    }

    # for 3 node quadtratic

    one_dimention_size = 5
    one_dim_el = one_dimention_size - 1
    amt_points = one_dimention_size**3
    layer_point_size = one_dimention_size**2
    # make dictionary of elements
    element_connectivity = []
    for i in range(one_dim_el):
        for j in range(one_dim_el):
            for k in range(one_dim_el):
                element_connectivity.append(
                    np.array(
                        [
                            i * one_dimention_size + j + k * layer_point_size,
                            i * one_dimention_size + j + 1 + k * layer_point_size,
                            i * one_dimention_size + j + one_dimention_size + 1 + k * layer_point_size,
                            i * one_dimention_size + j + one_dimention_size + k * layer_point_size,
                            i * one_dimention_size + j + (k + 1) * layer_point_size,
                            i * one_dimention_size + j + 1 + (k + 1) * layer_point_size, 
                            i * one_dimention_size + j + one_dimention_size + 1 + (k + 1) * layer_point_size,
                            i * one_dimention_size + j + one_dimention_size + (k + 1) * layer_point_size,
                        ],
                        dtype=int,
                    )
                )  # fmt: skip

    element_connectivity = {
        "hexahedron": np.array(element_connectivity),
    }

    x = np.linspace(0, prop.w, one_dimention_size)
    y = np.linspace(0, prop.h, one_dimention_size)
    z = np.linspace(0, prop.d, one_dimention_size)
    x, y, z = np.meshgrid(x, y, z)

    # flag the essential boundary conditions
    bc_fix_list = np.zeros(
        (one_dimention_size, one_dimention_size, one_dimention_size), dtype=int
    )
    # set boundy values to one since they are constrained
    bc_fix_list[0, :, :] = 1
    bc_fix_list[one_dimention_size - 1, :, :] = 1
    bc_fix_list[:, 0, :] = 1
    bc_fix_list[:, one_dimention_size - 1, :] = 1
    bc_fix_list[:, :, 0] = 1
    bc_fix_list[:, :, one_dimention_size - 1] = 1

    bc_g_list = np.zeros_like(bc_fix_list, dtype=float)

    # set boundy values to one since they are constrained
    bc_g_list[0, :] = BoundyConditions.front
    bc_g_list[one_dimention_size - 1, :] = BoundyConditions.back
    bc_g_list[:, 0] = BoundyConditions.left
    bc_g_list[:, one_dimention_size - 1] = BoundyConditions.right
    bc_g_list[:, :, 0] = BoundyConditions.bottom
    bc_g_list[:, :, one_dimention_size - 1] = BoundyConditions.top

    # handle edges
    bc_g_list[0, :, -1] = (BoundyConditions.front + BoundyConditions.top) / 2
    bc_g_list[0, -1, :] = (BoundyConditions.bottom + BoundyConditions.front) / 2
    bc_g_list[0, :, 0] = (BoundyConditions.front + BoundyConditions.bottom) / 2
    bc_g_list[0, 0, :] = (BoundyConditions.front + BoundyConditions.left) / 2
    bc_g_list[-1, :, -1] = (BoundyConditions.top + BoundyConditions.back) / 2
    bc_g_list[-1, :, 0] = (BoundyConditions.bottom + BoundyConditions.back) / 2
    bc_g_list[-1, -1, :] = (BoundyConditions.right + BoundyConditions.back) / 2
    bc_g_list[-1, 0, :] = (BoundyConditions.left + BoundyConditions.back) / 2
    bc_g_list[:, -1, 0] = (BoundyConditions.bottom + BoundyConditions.right) / 2
    bc_g_list[:, -1, -1] = (BoundyConditions.top + BoundyConditions.right) / 2
    bc_g_list[:, 0, -1] = (BoundyConditions.top + BoundyConditions.left) / 2
    bc_g_list[:, 0, 0] = (BoundyConditions.bottom + BoundyConditions.left) / 2

    # handle corners
    # front
    bc_g_list[0, -1, 0] = (
        BoundyConditions.right + BoundyConditions.bottom + BoundyConditions.front
    ) / 3

    bc_g_list[0, 0, 0] = (
        BoundyConditions.left + BoundyConditions.bottom + BoundyConditions.front
    ) / 3

    bc_g_list[-1, -1, 0] = (
        BoundyConditions.right + BoundyConditions.bottom + BoundyConditions.back
    ) / 3

    bc_g_list[-1, 0, 0] = (
        BoundyConditions.left + BoundyConditions.bottom + BoundyConditions.back
    ) / 3

    # back
    bc_g_list[0, -1, -1] = (
        BoundyConditions.right + BoundyConditions.top + BoundyConditions.front
    ) / 3

    bc_g_list[0, 0, -1] = (
        BoundyConditions.left + BoundyConditions.top + BoundyConditions.front
    ) / 3

    bc_g_list[-1, -1, -1] = (
        BoundyConditions.right + BoundyConditions.top + BoundyConditions.back
    ) / 3

    bc_g_list[-1, 0, -1] = (
        BoundyConditions.left + BoundyConditions.top + BoundyConditions.back
    ) / 3

    # reshape to be 1=d
    bc_fix_list = bc_fix_list.reshape((1, amt_points))
    bc_g_list = bc_g_list.reshape((1, amt_points))
    return (
        x,
        y,
        z,
        element_connectivity,
        bc_fix_list,
        bc_g_list,
        quad_rules,
        one_dimention_size,
    )


class BoundyConditions:
    back = 100
    front = 50
    left = 75
    right = 25
    bottom = 0
    top = 0


(
    x,
    y,
    z,
    element_connectivity,
    bc_fix_list,
    bc_g_list,
    quad_rules,
    one_dimention_size,
) = build()

plt = plot_mesh(x, y, z, element_connectivity["hexahedron"])
plt.savefig("img/mesh_bc.png")

plt = plot_bc(x, y, z, bc_g_list)
plt.savefig("img/boundrys_bc.png")

mesh = build_mesh(
    x.reshape((-1,)),
    y.reshape((-1,)),
    z.reshape((-1,)),
    element_connectivity,
    1,
    bc_fix_list,
    bc_g_list,
)


def f(x, y, z):
    return 0


# %% Assemble the global stiffness matrix
K = assemble_stiffness(mesh, prop, quad_rules)

# %% Assemble the global right-hand-side force vector
F = assemble_rhs(mesh, f, quad_rules)

# load the known solutions into the solution matrix
solution = np.zeros(mesh.num_nodes * mesh.num_dof_per_node)
idx = np.where(bc_fix_list == 1)
solution[mesh.ID[idx]] = bc_g_list[idx]

# %% Solve
r1 = mesh.free_range
r2 = mesh.freefix_range
solution[r1] = np.linalg.solve(
    K[np.ix_(r1, r1)], F[r1] - K[np.ix_(r1, r2)] @ solution[r2]
)


def convert_sol_to_two_d(solution):
    # have to unpack to solution into the free range
    twod_sol = bc_g_list.reshape(
        one_dimention_size, one_dimention_size, one_dimention_size
    )
    free_range_vals = iter(solution[r1])
    for i in range(1, one_dimention_size - 1):
        for j in range(1, one_dimention_size - 1):
            for k in range(1, one_dimention_size - 1):
                twod_sol[i, j, k] = next(free_range_vals)
    return twod_sol


unpacked_solution = convert_sol_to_two_d(solution)

plt = plot_solution(unpacked_solution, x, y)
plt.savefig("img/contour_bc.png")
