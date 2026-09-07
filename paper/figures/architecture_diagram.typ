#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#set page(width: auto, height: auto, margin: 6mm, fill: white)
#set text(size: 8pt, font: "New Computer Modern")

#let box(pos, label, fill: rgb("#eef4ff"), stroke: rgb("#33507a"), ..args) = node(
  pos, align(center, label),
  width: 34mm, height: 9mm,
  fill: fill, stroke: 0.9pt + stroke,
  corner-radius: 3pt,
  ..args,
)

#diagram(
  spacing: 7pt,
  cell-size: (8mm, 11mm),
  edge-stroke: 0.9pt,
  mark-scale: 65%,

  // Main vertical flow
  box((0,0), [Query / Prompt], fill: rgb("#f3f7ff")),
  box((0,1), [Task Router], fill: rgb("#fff4d6"), stroke: rgb("#8a6d1a")),
  box((0,4), [PLE Projector], fill: rgb("#f2e8fb"), stroke: rgb("#6b3a9b")),
  box((0,5), [Token Policy], fill: rgb("#fff0e6"), stroke: rgb("#b05a1a")),
  box((0,6), [Logit Fusion], fill: rgb("#f0f0f0"), stroke: rgb("#444444")),
  box((0,7), [Output], fill: rgb("#eaeaea"), stroke: rgb("#222222")),

  // Three complementary channels
  box((1,1), [BM25 / RAG], fill: rgb("#e7f6ec"), stroke: rgb("#1a6b3a")),
  box((1,2), [PLE / Engram], fill: rgb("#e8f0fe"), stroke: rgb("#1a4b8a")),
  box((1,3), [Backbone + MoRA], fill: rgb("#fdecee"), stroke: rgb("#9b2c3a")),

  // Main vertical edges
  edge((0,0), (0,1), "-|>"),
  edge((0,4), (0,5), "-|>"),
  edge((0,5), (0,6), "-|>"),
  edge((0,6), (0,7), "-|>"),

  // Router to channels
  edge((0,1), (1,1), "-|>"),
  edge((0,1), (1,2), "-|>"),
  edge((0,1), (1,3), "-|>"),

  // Channels to Projector
  edge((1,1), (0,4), "-|>", bend: 30deg),
  edge((1,2), (0,4), "-|>"),
  edge((1,3), (0,4), "-|>", bend: -30deg),
)
