#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#set page(width: auto, height: auto, margin: 6mm, fill: white)
#set text(size: 8pt, font: "New Computer Modern")

#let box(pos, label, fill: rgb("#eef4ff"), stroke: rgb("#33507a"), ..args) = node(
  pos, align(center, label),
  width: 30mm, height: 8mm,
  fill: fill, stroke: 0.9pt + stroke,
  corner-radius: 3pt,
  ..args,
)

#let panel(pos, label, fill, stroke) = {
  node(enclose: pos, fill: fill, stroke: 0.8pt + stroke, inset: 5pt, corner-radius: 5pt, name: label)
}

#diagram(
  spacing: 7pt,
  cell-size: (10mm, 11mm),
  edge-stroke: 0.9pt,
  mark-scale: 65%,

  panel(((0,0), (0,1)), <input>, rgb("#f8f9fb"), rgb("#8a8f98")),
  panel(((1,2), (1,4)), <evidence>, rgb("#f7fbf8"), rgb("#7a9b84")),
  panel(((0,6), (0,9)), <fusion>, rgb("#fbf8fc"), rgb("#9b7ab5")),

  box((0,0), [Query / Prompt], fill: rgb("#f3f7ff")),
  box((0,1), [Task Router], fill: rgb("#fff4d6"), stroke: rgb("#8a6d1a")),

  box((1,2), [BM25 / RAG], fill: rgb("#e7f6ec"), stroke: rgb("#1a6b3a")),
  box((1,3), [PLE / Engram], fill: rgb("#e8f0fe"), stroke: rgb("#1a4b8a")),
  box((1,4), [Backbone + MoRA], fill: rgb("#fdecee"), stroke: rgb("#9b2c3a")),

  box((2,3), [Context\ Features], fill: rgb("#eef4ff"), stroke: rgb("#33507a")),

  box((0,6), [PLE Projector], fill: rgb("#f2e8fb"), stroke: rgb("#6b3a9b")),
  box((0,7), [Token Policy], fill: rgb("#fff0e6"), stroke: rgb("#b05a1a")),
  box((0,8), [Logit Fusion], fill: rgb("#f0f0f0"), stroke: rgb("#444444")),
  box((0,9), [Output], fill: rgb("#eaeaea"), stroke: rgb("#222222")),

  edge((0,0), (0,1), "-|>"),
  edge((0,1), (1,2), "-|>"),
  edge((0,1), (1,3), "-|>"),
  edge((0,1), (1,4), "-|>"),

  edge((1,2), (2,3), "-|>"),
  edge((1,3), (2,3), "-|>"),
  edge((1,4), (2,3), "-|>"),

  edge((2,3), (0,6), "-|>", bend: -25deg),

  edge((0,6), (0,7), "-|>"),
  edge((0,7), (0,8), "-|>"),
  edge((0,8), (0,9), "-|>"),
)
