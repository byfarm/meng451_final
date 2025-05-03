#set page(paper: "us-letter", margin: (x: 0.5in, y: 0.5in))
#set heading(numbering: none)
#set enum(numbering: "1.", spacing: 1em)
#set math.equation(numbering: "(1)", block: true)
#set image(width: 60%)

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

= 3-D Figures 
#figure(image("img/mesh_bc.png"), caption: [3-D Mesh])
= Convection Figures
#figure(image("img/boundrys_2d.png"))
