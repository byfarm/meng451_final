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

