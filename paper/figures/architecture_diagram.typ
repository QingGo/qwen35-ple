#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#set page(width: auto, height: auto, margin: 10mm, fill: white)
#set text(size: 9.5pt, font: "New Computer Modern")

#let box(pos, label, fill: rgb("#eef4ff"), stroke: rgb("#33507a"), ..args) = node(
  pos, align(center, label),
  width: 34mm, height: 11mm,
  fill: fill, stroke: 0.9pt + stroke,
  corner-radius: 3pt,
  ..args,
)

#let panel(pos, label, fill, stroke) = {
  node(enclose: pos, fill: fill, stroke: 0.8pt + stroke, inset: 6pt, corner-radius: 5pt, name: label)
}

#let panel_label(pos, label) = node(
  pos, align(center, text(size: 8.5pt, weight: "bold", label)),
  width: auto, height: auto, fill: none, stroke: none
)

#diagram(
  spacing: 7pt,
  cell-size: (10mm, 11mm),
  edge-stroke: 0.9pt,
  mark-scale: 65%,

  panel(((0,0), (0,1)), <input>, rgb("#f8f9fb"), rgb("#8a8f98")),
  panel(((1,2), (1,4)), <evidence>, rgb("#f7fbf8"), rgb("#7a9b84")),
  panel(((0,6), (0,9)), <fusion>, rgb("#fbf8fc"), rgb("#9b7ab5")),

  panel_label((0,-0.55), [Input & Routing]),
  panel_label((1,1.40), [Evidence Channels]),
  panel_label((0,5.40), [Fusion & Generation]),

  box((0,0), [Query / Prompt], fill: rgb("#f3f7ff")),
  box((0,1), [Task Router], fill: rgb("#fff4d6"), stroke: rgb("#8a6d1a")),

  box((1,2), [BM25 / RAG\ document evidence], fill: rgb("#e7f6ec"), stroke: rgb("#1a6b3a")),
  node((1,3), align(center)[PLE / Engram\ n-gram memory],
    shape: fletcher.shapes.cylinder,
    width: 33mm, height: 12mm,
    fill: rgb("#e8f0fe"), stroke: 0.9pt + rgb("#1a4b8a")),
  box((1,4), [Backbone + MoRA\ parametric adapter], fill: rgb("#fdecee"), stroke: rgb("#9b2c3a")),

  box((2,3), [Context Features\ hidden + memory feats], fill: rgb("#eef4ff"), stroke: rgb("#33507a")),

  box((0,6), [PLE Projector\ $alpha_t, beta_t$], fill: rgb("#f2e8fb"), stroke: rgb("#6b3a9b")),
  box((0,7), [Token Policy\ safety gate], fill: rgb("#fff0e6"), stroke: rgb("#b05a1a")),
  box((0,8), [Logit Fusion\ calibrated logits], fill: rgb("#f0f0f0"), stroke: rgb("#444444")),
  box((0,9), [Output\ generated text], fill: rgb("#eaeaea"), stroke: rgb("#222222")),

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
