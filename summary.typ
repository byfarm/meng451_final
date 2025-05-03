#set page(paper: "us-letter", margin: (x: 0.5in, y: 0.5in))
#set heading(numbering: none)
#set enum(numbering: "1.", spacing: 1em)
#set math.equation(numbering: "(1)", block: true)

#let homework_header(title, class, professor, author: [Byron Farmar]) = {
  [
    #align(center, text(17pt)[#title])
    #align(center, [
      #author,
      #class,
      #professor
    ])
  ]
}

#homework_header("Final Project", "Gonzaga MENG 451", "Dr. Fitzgerald")

= Summary

The FEM program was first expanded to three dimentions to handle a convection problem with exclusively essential boundry conditions. The program was then expanded to handle a convection boundry condition in one dimention. Finally, both parts were combined and a 3D problem with a 2D convection boundry condition was solved.

In @final_bc, the boundry contitions for the back of the cube was set at 100, while the sides were set to 20. The front had the convection boundry condition applied. The dimentions of the cube were 1x1x1, the convection coefficient was set to 100, and the conduction coefficient was set to 273.

As can be seen in @final_results, the results seem to converge as the number of elements in the mesh increase.

= Figures
#figure(image("img/mesh_all.png"), caption: [Final Mesh Visualization. An additional face was applied across the front face (like seen in @final_bc) to apply the convection boundry condition.]) <final_mesh>
#figure(image("img/boundrys_all.png"), caption: [Final Boundry Conditions. The red wiring denotes the location of the convection boundry condition.]) <final_bc>
#figure([#image("img/contour_all_size_3.png", width: 80%) #image("img/contour_all_size_5.png", width: 80%) #image("img/contour_all_size_7.png", width: 100%) #image("img/contour_all_size_9.png", width: 100%)], caption: [Final Mesh Visualization. Note that the white dots are the location of the nodes at each layer. Layers are symetric so only half of the slices are shown except for the 3x3 mesh. Results are for 3, 5, 7, and 9 node deep meshes.]) <final_results>
