# Bundled fonts

The one binary in this repository, and it is here because no Debian font package the OCR
image installs carries the Tifinagh block. `fonts-noto-core` does not include it, so
`modal_app.common.ocr_image` mounts this directory instead and
`agbalu.ocr.synthetic.TIFINAGH_FONT_CANDIDATES` resolves it at `/root/resources/fonts/`.

| File | Family | Version | Copyright | Licence |
|---|---|---|---|---|
| `NotoSansTifinagh-Regular.ttf` | Noto Sans Tifinagh | 2.002 | Copyright 2019 Google Inc. | [SIL Open Font License 1.1](http://scripts.sil.org/OFL) |

Every field above is read from the file's own `name` table, not from the page it was
downloaded from.

**OFL-1.1 permits bundling and redistribution**, including inside a commercial work, on two
conditions this repository meets: the font is not sold on its own, and the licence travels
with it — which is what this file is. The font is not renamed, so the reserved-font-name
clause does not bind. Nothing in the OFL reaches the model weights: the licence covers the
font software, not images rendered with it, and `Feraoun-36M` is Apache-2.0.
