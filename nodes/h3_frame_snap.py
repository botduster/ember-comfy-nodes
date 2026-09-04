"""Ember H3 Frame Snap — pick a frame count MiniMax H3 can actually generate.

WHY THIS IS NOT A MATHS NODE. H3's video VAE runs on a 17k+5 grid: 5, 22, 39, 56, … 192,
209. An off-grid length is not rejected — it is silently realigned, and the TARGET and the
REFERENCE are realigned in opposite directions (target rounds up, reference rounds down).
At 198 frames that means a 209-frame video generated against 192 frames of reference: 17
frames, 0.71 s at 24 fps, with no reference conditioning at all. The model invents an ending.

On top of that, H3 clocks audio at 40 Hz against 24 fps video, so a frame boundary lands
exactly on the audio grid only when (frames * 40) % 24 == 0. Miss it and the audio drifts
against the picture over the tail, which is what breaks lip sync while the image still looks
fine.

Only lengths on BOTH grids are safe. Verified rather than assumed:

    17k+5 ∩ (frames*40 % 24 == 0)  →  39, 90, 141, 192, 243, 294 …   step 51

  and the algebra behind it: (17k+5) % 3 == 0  ⟺  k ≡ 2 (mod 3), which is why the safe runs
  sit 3 × 17 = 51 apart. Every safe length is divisible by 3.

That second constraint cannot be expressed in a generic maths node, which is the whole
reason this file exists.

Part of the Ember node pack. Reimplemented from the documented behaviour of H3's VAE and
audio clock — not derived from any third-party pack's source.
"""

# H3 video VAE grid: 5, 22, 39, … (17k + 5).
VIDEO_OFFSET = 5
VIDEO_STEP = 17
# The subset whose end lands exactly on the 40 Hz audio clock: 39, 90, 141, … (51k + 39).
AUDIO_OFFSET = 39
AUDIO_STEP = 51

MAX_FRAMES = 100_000
FPS = 24.0

MODE_AV = "Video + Audio (lip sync safe)"
MODE_V = "Video only (finer steps)"
MODES = [MODE_AV, MODE_V]

ROUND_NEAREST = "nearest"
ROUND_DOWN = "down"
ROUNDINGS = [ROUND_NEAREST, ROUND_DOWN]


def grid_for(alignment: str) -> tuple[int, int]:
    """(offset, step) for the requested grid."""
    if alignment == MODE_V:
        return VIDEO_OFFSET, VIDEO_STEP
    return AUDIO_OFFSET, AUDIO_STEP


def snap_frames(frame_count: int, alignment: str = MODE_AV, rounding: str = ROUND_NEAREST) -> int:
    """Nearest valid length at or below (or nearest to) `frame_count`.

    Never returns 0: below the first grid point there is no legal length, so the first one is
    returned rather than an empty video. A caller asking for 10 frames of an audio-aligned
    clip cannot have it, and 39 is the honest answer.
    """
    if frame_count < 1:
        return AUDIO_OFFSET if alignment == MODE_AV else VIDEO_OFFSET

    offset, step = grid_for(alignment)
    if frame_count <= offset:
        return offset

    k_down = (frame_count - offset) // step
    down = offset + k_down * step

    if rounding == ROUND_DOWN or down == frame_count:
        return down

    # nearest: consider the run above, and take it only if it is genuinely closer. Overshoot
    # is bounded by half a step, and the caller trims those frames — which is cheaper than
    # losing a full 51-frame run (over two seconds) for the sake of four missing frames.
    up = down + step
    return up if (up - frame_count) < (frame_count - down) else down


class EmberH3FrameSnap:
    """Snap a frame count onto a length MiniMax H3 can generate."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_count": (
                    "INT",
                    {
                        "default": 198,
                        "min": 1,
                        "max": MAX_FRAMES,
                        "step": 1,
                        "tooltip": "The frame count from your video loader.",
                    },
                ),
                "alignment": (
                    MODES,
                    {
                        "default": MODE_AV,
                        "tooltip": (
                            "Video + Audio: only lengths on H3's 17k+5 video grid that ALSO "
                            "land on its 40 Hz audio clock (39, 90, 141, 192, 243…). Use this "
                            "whenever the clip has speech — it is what keeps lip sync locked "
                            "to the last frame.\n"
                            "Video only: the full 17k+5 grid. Finer steps, but audio can drift "
                            "over the tail."
                        ),
                    },
                ),
                "rounding": (
                    ROUNDINGS,
                    {
                        "default": ROUND_NEAREST,
                        "tooltip": (
                            "nearest: closest valid run either way; overshoot is at most half a "
                            "step and those frames are trimmed.\n"
                            "down: never exceed the source. Safer on paper, but missing a run "
                            "by four frames costs a full 51-frame step."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("frames", "seconds")
    FUNCTION = "snap"
    CATEGORY = "Ember/H3"
    DESCRIPTION = (
        "Snap a frame count onto a length MiniMax H3 can generate, keeping the target and "
        "the reference on the same grid so the tail is not left unconditioned."
    )

    def snap(self, frame_count: int, alignment: str, rounding: str):
        frames = snap_frames(int(frame_count), alignment, rounding)
        return (frames, frames / FPS)


NODE_CLASS_MAPPINGS = {"EmberH3FrameSnap": EmberH3FrameSnap}
NODE_DISPLAY_NAME_MAPPINGS = {"EmberH3FrameSnap": "Ember H3 Frame Snap"}
