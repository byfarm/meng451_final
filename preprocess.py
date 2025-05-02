import numpy as np


class Mesh:
    def __init__(
        self,
        x,
        y,
        z,
        IEN,
        ID,
        LM,
        ned,
        neq,
        nnp,
        ng,
        BC_g_list,
        BC_fix_list,
        free_range,
        freefix_range,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.element_connectivity = IEN
        self.ID = ID
        self.LM = LM
        self.num_dof_per_node = ned
        self.num_free_equations = neq
        self.num_nodes = nnp
        self.num_constrained_corrds = ng
        self.BC_g_list = BC_g_list
        self.BC_fix_list = BC_fix_list
        self.free_range = free_range
        self.freefix_range = freefix_range


class ElementDef:
    def __init__(self, name, nen, internal_dim, gmsh_number):
        self.name = name
        self.num_nodes_in_element = nen
        self.internal_dim = internal_dim
        self.gmsh_number = gmsh_number


# Standard GMSH element definitions
std_element_defs = {
    "line": ElementDef("Linear Line", 2, 1, 1),
    "curve": ElementDef("3 node quadratic", 3, 1, 1),
    "triangle": ElementDef("Linear Triangle", 3, 2, 2),
    "quad": ElementDef("4 node quadrilateral", 4, 2, 3),
    "hexahedron": ElementDef("8 node hexahedron", 8, 3, 3),
}


def build_identity_matrix(num_nodes, g_list, dof_per_node, fix_list):
    number_of_free_equations = 0
    count = 0

    # Compute ng
    num_constrained_coords = np.sum(fix_list > 0)

    # Construct ID
    identity_matrix = np.zeros((dof_per_node, num_nodes), dtype=int)
    totaldof = dof_per_node * num_nodes

    if num_constrained_coords > 0:
        count = 0
        for A in range(num_nodes):
            for i in range(dof_per_node):
                if fix_list[i, A] == 1:
                    identity_matrix[i, A] = totaldof - count
                    count += 1
                else:
                    number_of_free_equations += 1
                    identity_matrix[i, A] = number_of_free_equations
    else:
        for A in range(num_nodes):
            for i in range(dof_per_node):
                number_of_free_equations += 1
                identity_matrix[i, A] = number_of_free_equations

    # Fix numbering to be zero-based
    identity_matrix = identity_matrix - 1

    free_range = range(number_of_free_equations)
    freefix_range = range(
        number_of_free_equations, number_of_free_equations + num_constrained_coords
    )

    return (
        identity_matrix,
        number_of_free_equations,
        num_constrained_coords,
        free_range,
        freefix_range,
    )


def build_LM(identity_matrix, IEN):
    # lm you are trying to match each degree of freedom with the element
    ned = identity_matrix.shape[0]
    LM = {}

    for element_type, ien in IEN.items():
        num_elements = ien.shape[0]
        num_nodes_in_element = std_element_defs[element_type].num_nodes_in_element
        lm = np.zeros((num_nodes_in_element * ned, num_elements), dtype=int)

        for e in range(num_elements):
            for a in range(num_nodes_in_element):
                for i in range(ned):
                    p = a + num_nodes_in_element * (i)
                    lm[p, e] = identity_matrix[i, ien[e, a]]

        LM[element_type] = lm

    return LM


def build_mesh(x, y, z, element_connectivity, DofPerNode, BC_fix_list, BC_g_list):
    num_nodes = len(x)
    ID, num_free_equations, num_constrained_coords, free_range, freefix_range = (
        build_identity_matrix(num_nodes, BC_g_list, DofPerNode, BC_fix_list)
    )
    LM = build_LM(ID, element_connectivity)

    return Mesh(
        x,
        y,
        z,
        element_connectivity,
        ID,
        LM,
        DofPerNode,
        num_free_equations,
        num_nodes,
        num_constrained_coords,
        BC_g_list,
        BC_fix_list,
        free_range,
        freefix_range,
    )
