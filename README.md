# Ember ComfyUI nodes

Ember's own ComfyUI nodes, rebuilt so a workflow can be shared without the recipient
needing anyone else's licensed pack installed.

## Krea 2 workflow nodes (added 2026-09-15)

Six nodes that replace the AIORBust nodes in the Krea 2 workflows. Widget names, order,
defaults and limits match, so in a UI workflow you can change a node's `type` to the
Ember id and keep its `widgets_values`. Proven bit-identical to the originals on CPU and
on a GPU with the real Krea 2 model (evidence: `botduster/krea2-t2i-serverless`,
`tests/ember_nodes/`). Full widget table: `NODE_SPEC.md`.

| Ember node (menu `Ember/…`) | Replaces |
|---|---|
| Ember Resolution (MP) `EmberResolutionMP` | Aiorbust Resolution (MP) |
| Ember Save Image (No Metadata) `EmberSaveImageNoMetadata` | Aiorbust Save Image No Metadata |
| Ember Camera Look `EmberCameraLook` | Aiorbust Camera Look |
| Ember Renoise `EmberRenoise` | Aiorbust Renoïse |
| Ember HD Ultralytics BBox Loader `EmberHDBBoxDetectorProvider` | Aiorbust HD Ultralytic BBox Loader (bbox/ models only) |
| Ember Detailer `EmberDetailer` | Aiorbust Detailer and Aiorbust Eye Detailer |

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

Update later with `cd "$C/custom_nodes/ember-comfy-nodes" && git pull`.

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
