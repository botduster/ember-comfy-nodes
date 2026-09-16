# Ember ComfyUI nodes

Ember's own ComfyUI nodes, rebuilt so a workflow can be shared without the recipient
needing anyone else's licensed pack installed.

## Krea 2 workflow nodes (added 2026-09-15)

Seven nodes that replace the AIORBust nodes in the Krea 2 workflows. Widget names, order,
defaults and limits match, so in a UI workflow you can change a node's `type` to the
Ember id and keep its `widgets_values`. The first six are proven bit-identical to the
originals on CPU and on a GPU with the real Krea 2 model (evidence:
`botduster/krea2-t2i-serverless`, `tests/ember_nodes/`). Ember Speed HD Sampler is proven
bit-identical on CPU (`tests/speed_hd_sampler/`, below); its GPU run with the real model is
still to do. Full widget table: `NODE_SPEC.md`.

| Ember node (menu `Ember/…`) | Replaces |
|---|---|
| Ember Resolution (MP) `EmberResolutionMP` | Aiorbust Resolution (MP) |
| Ember Save Image (No Metadata) `EmberSaveImageNoMetadata` | Aiorbust Save Image No Metadata |
| Ember Camera Look `EmberCameraLook` | Aiorbust Camera Look |
| Ember Renoise `EmberRenoise` | Aiorbust Renoïse |
| Ember HD Ultralytics BBox Loader `EmberHDBBoxDetectorProvider` | Aiorbust HD Ultralytic BBox Loader (bbox/ models only) |
| Ember Detailer `EmberDetailer` | Aiorbust Detailer and Aiorbust Eye Detailer |
| Ember Speed HD Sampler `EmberSpeedHDSampler` (added 2026-09-16) | Aiorbust Speed HD Sampler |

**No AIORBust node left in the Krea V3 T2I graph.** In the V3 graph as run on 2026-09-16
(already using the six nodes above), `AiorbustSpeedHDSampler` was the only AIORBust class.
With Ember Speed HD Sampler swapped in, that graph no longer needs
`public-aiorbust-nodes-pack`. It still needs the other packs it uses (RvTools, CRT,
FameGridColorFinish, RealismInjector, SkinDetailer, MoreJPEG, LayerStyle, KJNodes, rgthree).

**Swapping in Ember Speed HD Sampler.** It sits inside the "Speed Spectral Diff. High Res"
subgraph, so edit the node there. Change `type` from `AiorbustSpeedHDSampler` to
`EmberSpeedHDSampler` and keep all 11 `widgets_values`: the 10 widgets, plus
`control_after_generate` after `seed`. The links to `base_sampler` and `spectrum_beta` stay
as they are. In an API graph, change `class_type`; the inputs do not change. `transform = dwt`
needs PyWavelets, which is in `requirements.txt`. The noise added when the latent grows is
computed in numpy on the CPU, the same as the original.

The pack also registers the `beta57` scheduler, so RES4LYF is not needed for these graphs.
Known behaviour kept from the originals: Camera Look noise above 0 is unseeded; Renoise
grain differs between GPU and CPU for the same seed.

**Live preview on Ember Resolution (MP)** (`js/ember_resolution_mp.js`): a read-only line
under the widgets, in the same format as the AIORBust node:
`1600 x 2784  ·  4.454 MP (+0.1%)  ·  0.5747`. It updates as megapixels, aspect_ratio or
multiple_of change. When the ratio comes from the image, or a widget is driven by a link,
it says "computed at run time" instead of guessing. The line is never saved: the node
still stores exactly 3 `widgets_values` and sends 3 inputs. The browser maths is tested
against the Python node over 79,464 widget combinations with 0 differences, and 6 planted
bugs are all caught (`tests/resolution_preview/run.sh`, needs python3 + node).

⚠️ `EmberResolutionMP` changed on 2026-09-15: it now matches Aiorbust Resolution (MP)
exactly (the earlier algorithm differed on ~19% of sizes and lacked 5:4 / 4:5).

### Install on a RunPod pod

```bash
C=/workspace/ComfyUI; [ -d "$C" ] || C=/ComfyUI     # /workspace only when a network volume is attached
cd "$C/custom_nodes"
git clone https://github.com/botduster/ember-comfy-nodes.git
[ -d ComfyUI-Impact-Pack ] || git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git   # needed by Ember Detailer
python3 -m pip install -r ember-comfy-nodes/requirements.txt       # use the Python ComfyUI runs with (e.g. /opt/venv)
python3 -m pip install -r ComfyUI-Impact-Pack/requirements.txt
mkdir -p "$C/models/ultralytics/bbox"   # Eyeful_v2-Paired.pt goes here (the file the AIORBust eye detailer used)
# then restart ComfyUI
```

Bit-identical output was proven on ComfyUI v0.27.0 + Impact Pack 8.28.3. On newer stacks, run one
workflow with the original and Ember nodes side by side once to confirm.

Update later with `cd "$C/custom_nodes/ember-comfy-nodes" && git pull`, then restart ComfyUI.

### Ember Speed HD Sampler: CPU equivalence test

```bash
git clone https://github.com/aiorbust-jy/public-aiorbust-nodes-pack && git -C public-aiorbust-nodes-pack checkout 0ffd558
PYTHON=/path/to/comfyui/python tests/speed_hd_sampler/run.sh <ComfyUI dir> public-aiorbust-nodes-pack --mutants
```

Both nodes run in one process, through the same call that SamplerCustomAdvanced makes, with a
small deterministic fake model. Compared exactly (dtype, shape, stride, `torch.equal` and
byte for byte):
- every latent and sigma slice passed to the base solver, including the grown latent with its
  noise and the aligned sigma;
- every model call and progress callback;
- the sampled latent;
- the global RNG states;
- raised exceptions.

Coverage:
- Wyatt's V3 graph settings on the full 1600x2784 latent.
- Every transform x mode x preset x 5 scale lists x 4 latent shapes.
- Edge values of delta, spectrum_A/beta, seed, scales and manual_sigmas (including invalid ones).
- 7 schedules.
- All 42 base solvers.
- 3 dtypes.

The reference also runs twice, as a noise floor. `mutants.py` plants 18 single-point bugs,
and the test must catch each one.

Result on 2026-09-16 (ComfyUI v0.34.6, torch 2.14.0, numpy 2.2.6, scipy 1.15.3, PyWavelets
1.8.0, the pod's versions): 755 checks, 0 mismatches, floor 0, 18/18 mutants caught
(`tests/speed_hd_sampler/results/`).

## Utility nodes

| Node | Does |
|---|---|
| **Ember H3 Frame Snap** | Snaps a frame count onto a length MiniMax H3 accepts |
| **Ember Audio Switch** | Include or drop an audio track without rewiring |
| **Ember Video Frame** | One frame out of a video file or a connected batch |

## Install

Clone or copy into `ComfyUI/custom_nodes/` and restart. Requires `av` only for the
file path of Ember Video Frame; everything else is numpy/torch, which ComfyUI has.

## Why this exists

A graph that references a licensed pack cannot be handed to a teammate or shipped
to a customer — they need the licence too, even for nodes that do nothing but
arithmetic. These four were the ones actually in use.

## The one with real content

**H3 Frame Snap** encodes MiniMax H3's accepted clip lengths:

- **Video grid** — `17k + 5`: 5, 22, 39, 56 …
- **Audio-aligned grid** — `51k + 39`: 39, 90, 141, 192 … the subset that also lands
  on H3's 40 Hz audio clock. Use it whenever the clip has speech; off it, audio
  drifts against picture over the tail and lip sync fails at the END of the clip.
- **Ceiling** — 3600 frames.

It rounds **up** by default. Rounding down looks safer and is worse: the audio grid
steps by 51 frames (~2 s at 24 fps), so a clip four frames above a valid length
loses two full seconds. Rounding up overshoots by at most half a step, and those
frames sit past the reference unconditioned, so they get trimmed.

## Not a drop-in rename

These are reimplementations from observed behaviour, not copies. Names and
categories differ deliberately (`Ember/*`), so both packs can be installed at once
while a workflow is migrated node by node.
