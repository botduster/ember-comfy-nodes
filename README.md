# Ember ComfyUI nodes

Four utility nodes, rebuilt so a workflow can be shared without the recipient
needing anyone else's licensed pack installed.

| Node | Does |
|---|---|
| **Ember H3 Frame Snap** | Snaps a frame count onto a length MiniMax H3 accepts |
| **Ember Resolution (MP)** | Megapixel budget + aspect ratio → snapped width/height |
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
