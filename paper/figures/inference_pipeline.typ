#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import fletcher.shapes: diamond
#set page(width: auto, height: auto, margin: 6mm, fill: white)
#set text(size: 8pt, font: "New Computer Modern")

#let step(pos, label, fill: rgb("#eef4ff"), stroke: rgb("#33507a"), ..args) = node(
  pos, align(center, label),
  width: 36mm, height: 8mm,
  fill: fill, stroke: 0.9pt + stroke,
  corner-radius: 3pt,
  ..args,
)

#diagram(
  spacing: 6pt,
  cell-size: (9mm, 10mm),
  edge-stroke: 0.9pt,
  mark-scale: 65%,

  step((0,0), [Query / Prompt], fill: rgb("#f3f7ff")),
  step((0,1), [Task Classifier], fill: rgb("#fff4d6"), stroke: rgb("#8a6d1a")),
  step((0,2), [BM25 + PLE\ Retrieval], fill: rgb("#e7f6ec"), stroke: rgb("#1a6b3a")),
  step((0,3), [Memory Feature\ Extraction], fill: rgb("#e8f0fe"), stroke: rgb("#1a4b8a")),

  node((0,4), align(center)[Token Policy:\ confident PLE evidence?],
    shape: diamond, width: 38mm, height: 13mm,
    fill: rgb("#fff0e6"), stroke: 0.9pt + rgb("#b05a1a")),

  step((0,5), [PLE Projector\ -> scale / bias], fill: rgb("#f2e8fb"), stroke: rgb("#6b3a9b")),
  step((1,5), [Base Logits\ only], fill: rgb("#f0f0f0"), stroke: rgb("#444444")),
  step((0,6), [Logit Fusion\ + Safety Gate], fill: rgb("#f0f0f0"), stroke: rgb("#444444")),
  step((0,7), [Decode / Generated Text], fill: rgb("#eaeaea"), stroke: rgb("#222222")),

  edge((0,0), (0,1), "-|>"),
  edge((0,1), (0,2), "-|>", [1. retrieve]),
  edge((0,2), (0,3), "-|>"),
  edge((0,3), (0,4), "-|>"),

  edge((0,4), (0,5), "-|>", [yes]),
  edge((0,4), (1,5), "-|>", [no]),
  edge((0,5), (0,6), "-|>"),
  edge((1,5), (0,6), "-|>", bend: -30deg),
  edge((0,6), (0,7), "-|>"),
)
