#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import fletcher.shapes: diamond
#set page(width: auto, height: auto, margin: 6mm, fill: white)
#set text(size: 8pt, font: "New Computer Modern")

#let step(pos, label, fill: rgb("#eef4ff"), stroke: rgb("#33507a"), ..args) = node(
  pos, align(center, label),
  width: 34mm, height: 8mm,
  fill: fill, stroke: 0.9pt + stroke,
  corner-radius: 3pt,
  ..args,
)

#let panel(pos, fill, stroke) = {
  node(enclose: pos, fill: fill, stroke: 0.8pt + stroke, inset: 5pt, corner-radius: 5pt)
}

#diagram(
  spacing: 7pt,
  cell-size: (10mm, 11mm),
  edge-stroke: 0.9pt,
  mark-scale: 65%,

  panel(((0,0), (0,3)), rgb("#f6f9fc"), rgb("#7a9bb5")),
  panel(((0,6), (1,8)), rgb("#f9f6fc"), rgb("#9b7ab5")),

  step((0,0), [Query / Prompt], fill: rgb("#f3f7ff")),
  step((0,1), [Task Classifier], fill: rgb("#fff4d6"), stroke: rgb("#8a6d1a")),
  step((0,2), [BM25 + PLE\ Retrieval], fill: rgb("#e7f6ec"), stroke: rgb("#1a6b3a")),
  step((0,3), [Memory Feature\ Extraction], fill: rgb("#e8f0fe"), stroke: rgb("#1a4b8a")),

  node((0,4), align(center)[Token Policy:\ confident PLE evidence?],
    shape: diamond, width: 36mm, height: 13mm,
    fill: rgb("#fff0e6"), stroke: 0.9pt + rgb("#b05a1a")),

  step((0,6), [PLE Projector\ -> scale / bias], fill: rgb("#f2e8fb"), stroke: rgb("#6b3a9b")),
  step((1,6), [Base Logits\ only], fill: rgb("#f0f0f0"), stroke: rgb("#444444")),
  step((0,7), [Logit Fusion\ + Safety Gate], fill: rgb("#f0f0f0"), stroke: rgb("#444444")),
  step((0,8), [Decode / Generated Text], fill: rgb("#eaeaea"), stroke: rgb("#222222")),

  edge((0,0), (0,1), "-|>"),
  edge((0,1), (0,2), "-|>", [1. retrieve]),
  edge((0,2), (0,3), "-|>"),
  edge((0,3), (0,4), "-|>"),

  edge((0,4), (0,6), "-|>"),
  edge((0,4), (1,6), "-|>", [no], label-pos: 0.35),
  edge((0,6), (0,7), "-|>"),
  edge((1,6), (0,7), "-|>", bend: -35deg),
  edge((0,7), (0,8), "-|>"),
)
