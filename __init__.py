"""Ember ComfyUI nodes.

Plain-Python utility nodes with no licence check, no phone-home and no account.
They exist so a workflow built here can be handed to someone else and just run —
a graph that references a licensed third-party pack is not portable, however
harmless the individual nodes are.

Nothing here calls out to a network at RUN time. One exception, argued and
documented rather than slipped in: Ember Face Mask fetches OpenCV's YuNet weights
(~350KB) the first time it runs and caches them thereafter. Point its model_path at
a local file and even that stops.
"""

import logging

from .nodes import h3_frame_snap, audio_switch, video_frame, face_mask
# Krea 2 workflow nodes (2026-09-15): Ember's own versions of the six AIORBust nodes, proven
# bit-identical on CPU and GPU (evidence: botduster/krea2-t2i-serverless tests/ember_nodes).
# EmberResolutionMP now lives here and matches "Aiorbust Resolution (MP)" exactly; the older
# nodes/resolution_mp.py algorithm was removed so the id means one thing.
from .ember_nodes import (
    bbox_detector,
    camera_look,
    detailer,
    renoise,
    resolution_mp,
    save_image_no_metadata,
)
# Krea V3 (2026-09-16): Ember Speed HD Sampler replaces the last AIORBust node in the V3 graph.
from .ember_nodes import speed_hd_sampler

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Front-end extensions (the live preview line on Ember Resolution (MP)).
WEB_DIRECTORY = "./js"

for _mod in (h3_frame_snap, audio_switch, video_frame, face_mask,
             resolution_mp, save_image_no_metadata, camera_look, renoise, bbox_detector, detailer,
             speed_hd_sampler):
    NODE_CLASS_MAPPINGS.update(_mod.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_mod.NODE_DISPLAY_NAME_MAPPINGS)

try:
    from .ember_nodes.schedulers import register_beta57
    register_beta57()
except Exception as exc:  # a scheduler-registry change upstream must not take the whole pack down
    logging.warning("[Ember nodes] could not register beta57: %s", exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
