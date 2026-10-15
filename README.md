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

## Krea 2 carousel nodes (added 2026-10-15)

Four nodes for the Krea 2 V3 carousel workflow: one master photo plus a pose list in, one edited
image per pose out (Krea 2 + the Krea 2 Identity Edit LoRA + `comfyui-krea2edit`).

**Clean-room.** These replace nodes from a proprietary pack whose licence forbids using its code or
prompts for a competing product. They were written from the spec below (inputs, outputs, documented
behaviour) using public methods and Ember's own wording. No code or prompt text from that pack was
copied, and the pack was never run. They are tested against this spec, not against the originals.

| Ember node (menu `Ember/…`) | Does |
|---|---|
| Ember Krea2 Pose List Prep `EmberKrea2PoseListPrep` | Validates and normalizes a pose list that Ember wrote |
| Ember Krea2 Carousel Director `EmberKrea2CarouselDirector` | One Identity Edit instruction per pose (LIST output) + manifest + warnings |
| Ember Skin Tone `EmberSkinTone` | Rotates skin hue onto the skin-tone line; luma untouched |
| Ember Clarity `EmberClarity` | Edge-aware midtone local contrast; colour ratios kept |

### Spec

**Pose List Prep.** No vision model runs on the GPU worker: the Ember app writes the poses (its own
model call on the master photo, or from pose photos the user uploads) and sends them as text.
- Input `pose_list`: one pose per line, or JSON (`["…"]`, `[{"pose": "…"}]`, `{"poses": […]}`).
  Bullets, numbering and "Slide 3:" labels are stripped, whitespace collapsed, blanks and
  case-insensitive duplicates dropped, each pose capped at 300 characters (cut on a word).
- `slide_count` 0 = one slide per pose (max 8); N = the first N. Asking for more than given is an
  error unless `if_too_few` = "use what is there".
- Outputs `pose_list` (one per line), `slide_count`, `report`.

**Carousel Director.** Pure string compiler, no model.
- Per pose: `[trigger, ]` "Show the same person from the reference photo in a new pose." + "Keep her
  <locks> unchanged." + a camera sentence + "New pose: <pose>." + optional shoot direction +
  optional extra text.
- `continuity`: balanced = face, hair, skin tone, body shape, outfit; strong adds accessories,
  makeup, location, lighting; maximum adds background details, colour grade, photo style.
- `camera`: auto (angle suits the pose), locked (same position, lens, framing), free.
- `edit_prompts` and `negative_prompts` (empty strings) are LIST outputs, so ComfyUI runs everything
  downstream once per slide. Wire them to `Krea2EditGroundedEncode.prompt` (positive / negative).
  `output_mode` "one slide" + `slide_index` re-rolls a single slide.
- `pose_list_in` / `slide_count_in` (links, from Pose List Prep) override the widgets.
  `master_image` is read only for its size (manifest) and to warn when missing.
- `slide_seeds` (LIST) = `seed` + slide number - 1: wire it to RandomNoise.noise_seed so each slide gets
  its own noise, and a one-slide re-roll with the same seed reproduces that slide (a new seed re-rolls it).
- Pose cleaning is idempotent, so the Director re-reading Prep's output always agrees with Prep's count.
- `manifest` (JSON: slides, locks, recommended graph settings) and `report` (warnings when a pose
  asks for something a lock forbids: outfit, location, hair, lighting, or a camera move under
  "locked"). Warnings never block.

**Skin Tone.** NTSC YIQ; skin of every complexion sits near the +I axis. Hue error to the target
(+I + `target_offset_deg`); skin weight in two passes (find where this image's skin sits, then
centre the hue window there) with chroma and luma gates, Gaussian-softened, optionally multiplied
by `mask`. Rotation `d' = d - strength*w*(m + uniformity*(d - m))` (m = the skin's mean error);
`saturation_fix` pulls skin chroma toward its mean. Only I/Q change. Pixels with zero weight come
back bit-identical; `strength` 0 returns the input exactly. Outputs `image`, `skin_mask`.

**Clarity.** Rec.709 luma; base = self-guided filter (He et al. 2010), radius =
`radius_pct`% of the shorter side, eps = `edge_preserve`; `L' = L + amount*(L - base)*(4L(1-L))^midtone_focus`;
RGB scaled by L'/L (colour ratios kept). `amount` 0 returns the input exactly; negative softens.

### Tests

```bash
PYTHON=/path/to/comfyui/python tests/carousel_nodes/run.sh --mutants
```

61 spec tests (interface, exact prompt wording, parsing edge cases, list-output shape, maths on
constructed YIQ colours, an independent re-implementation of the Clarity formula, exactness of
untouched pixels, batch independence, alpha, dtype). `mutants.py` plants 46 single-point bugs and the
tests must catch each one. Result on 2026-10-15: 61/61 pass on torch 2.10 and 2.14.1; 46/46 mutants
caught. The workflow that uses them validated on ComfyUI v0.34.6 with no missing nodes
(`botduster/krea2-t2i-serverless`, branch `feat/krea2-v3-carousel`). Not yet run on a GPU with the real model.

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
