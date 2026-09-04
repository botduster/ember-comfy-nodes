"""Ember ComfyUI nodes.

Plain-Python utility nodes with no licence check, no phone-home and no account.
They exist so a workflow built here can be handed to someone else and just run —
a graph that references a licensed third-party pack is not portable, however
harmless the individual nodes are.

Nothing here calls out to a network. If a node in this pack ever needs to, that
is worth arguing about first.
"""

from .nodes import h3_frame_snap, resolution_mp, audio_switch, video_frame

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _mod in (h3_frame_snap, resolution_mp, audio_switch, video_frame):
    NODE_CLASS_MAPPINGS.update(_mod.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_mod.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
