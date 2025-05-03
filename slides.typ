#import "@preview/touying:0.6.1": *
#import themes.simple: *

#show: simple-theme.with(aspect-ratio: "16-9")

= 3-D Mesh with 2-D Convection
By Byron Farmar

MENG 451 Final Project

== 1-D Convection

- Apply 1-D Convection Boundry condition to Problem 2.
- Have to create another part of the mesh with line elements.
- Have to apply the boundry condition to stiffness matrix.
$ K_e = integral_Omega h N^T N det(J) dif xi $
#figure(image("img/mesh_2d.png"), caption: [2-D Mesh with 1-D Convection.]) <2d_mesh>
#figure(image("img/boundrys_2d.png"), caption: [2-D Boundry Conditions]) <2d_boundrys>
#figure(image("img/contour_2d.png"), caption: [2-D Results]) <2d_result>

== 3-D Mesh

- Must convert $K_e$ to handle 3 dimentions.
- Have to set up mesh correctly.
- Add shape function for hexahedron.
#figure(image("img/mesh_3d.png"), caption: [3-D Mesh]) <3d_mesh>
#figure(image("img/boundrys_3d.png"), caption: [3-D Boundry Conditions]) <3d_boundrys>
#figure(image("img/contour_3d.png"), caption: [3-D Results]) <3d_result>

== 2-D Convection
- Remove appropriate boundry condition.
- Change convection from line to quadralateral elements.
- Change the modification of $K_e$ to work with quadralateral elements.
$ K_e = integral_Omega h N^T N det(J) dif xi dif eta $
$ det(J) = norm(harpoon(r_",ξ") times harpoon(r_",η")) $
#figure(image("img/boundrys_all.png"), caption: [3-D Convection Boundry Conditions]) <all_boundrys>
#figure(image("img/contour_all_size_7.png"), caption: [3-D Convection Results]) <all_result>
