import numpy as np
from quadrature import Quadrule
from shapefunctions import get_shape_func
from preprocess import std_element_defs  # Assuming this exists in Preprocess


def assemble_stiffness(mesh, properties, quad_rules):
    """Assemble the global stiffness matrix."""
    num_dof_per_node = mesh.num_dof_per_node
    totaldofs = num_dof_per_node * mesh.num_nodes
    K = np.zeros((totaldofs, totaldofs))

    # Loop over each element in the mesh
    for element_type, element_connectivity in mesh.element_connectivity.items():
        num_elements = element_connectivity.shape[0]
        num_nodes_in_element = std_element_defs[element_type].num_nodes_in_element
        total_num_dof = num_dof_per_node * num_nodes_in_element
        LM = mesh.LM[element_type]

        for element in range(num_elements):
            A = element_connectivity[element, :num_nodes_in_element]
            x, y, z = mesh.x[A], mesh.y[A], mesh.z[A]

            ke = element_stiffness(x, y, z, properties, element_type, quad_rules)

            # Assemble element stiffness into the global stiffness matrix
            for loop1 in range(total_num_dof):
                i = LM[loop1, element]
                for loop2 in range(total_num_dof):
                    j = LM[loop2, element]
                    K[i, j] += ke[loop1, loop2]

    return K


def element_stiffness(
    x_pnts,
    y_pnts,
    z_pnts,
    properties,
    element_type: str,
    quad_rules: dict[str, Quadrule],
):
    """Compute the element stiffness matrix for a 1D bar."""
    shapefoo = get_shape_func(element_type)

    running_sum = np.zeros((len(x_pnts), len(x_pnts)))
    for ξi, wi in quad_rules[element_type].iterator:
        for ξj, wj in quad_rules[element_type].iterator:
            for ξk, wk in quad_rules[element_type].iterator:
                # Evaluate the shape function
                _, Nξ, Nη, Nζ = shapefoo((ξi, ξj, ξk))

                shape_derivatives = np.array([Nξ, Nη, Nζ]).T
                pnt_mat = np.array([x_pnts, y_pnts, z_pnts])

                jacob = pnt_mat @ shape_derivatives
                detJ = np.linalg.det(jacob)

                B = np.linalg.solve(jacob, shape_derivatives.T)

                running_sum += properties.K * B.T @ B * detJ * wi * wj * wk

    return running_sum


def assemble_rhs(mesh, external_forcing, quad_rules):
    """Assemble the global right-hand-side force vector."""
    ned = mesh.num_dof_per_node
    totaldofs = ned * mesh.num_nodes
    F = np.zeros(totaldofs)

    # Loop over each element in the mesh
    for element_type, element_connectivity in mesh.element_connectivity.items():
        num_elements = element_connectivity.shape[0]
        shape_funcs = get_shape_func(element_type)
        element_quad_rule = quad_rules[element_type]
        num_nodes_in_element = std_element_defs[element_type].num_nodes_in_element
        total_num_dof = ned * num_nodes_in_element
        LM = mesh.LM[element_type]

        for element in range(num_elements):
            A = element_connectivity[element, :num_nodes_in_element]
            x, y, z = mesh.x[A], mesh.y[A], mesh.z[A]

            fe = element_forcing(
                x,
                y,
                z,
                shape_funcs,
                element_quad_rule,
                external_forcing,
            )

            # Assemble element force into the global force vector
            for loop1 in range(total_num_dof):
                i = LM[loop1, element]
                F[i] += fe[loop1]

    return F


def element_forcing(xe, ye, ze, N, element_quad_rule: Quadrule, external_forcing):
    """Compute the element force vector."""
    ned = 1
    nen = len(xe)
    nee = ned * nen
    fe = np.zeros(nee)

    # Integration loop
    for ξi, wi in element_quad_rule.iterator:
        for ξj, wj in element_quad_rule.iterator:
            for ξk, wk in element_quad_rule.iterator:
                # Evaluate the shape function
                Ne, Nξ, Nη, Nζ = N((ξi, ξj, ξk))

                # Evaluate the external loading at x(ξ)
                x = np.dot(Ne, xe)
                y = np.dot(Ne, ye)
                z = np.dot(Ne, ze)
                fext = external_forcing(x, y, z)

                # WARNING: check the shape of the shape_derivatives
                shape_derivatives = np.array([Nξ, Nη, Nζ]).T
                diff_arr = np.array([xe, ye, ze])

                jacob = diff_arr @ shape_derivatives
                detJ = np.linalg.det(jacob)

                # Integrate
                fe += Ne * fext * detJ * wi * wj * wk

    return fe
