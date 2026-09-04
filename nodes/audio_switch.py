"""Ember Audio Switch — cut an audio connection from a widget.

WHY A NODE AND NOT CTRL+B. ComfyUI's bypass forwards a node's input straight to its
matching output, so bypassing an audio node does NOT sever the audio — it still
arrives downstream. Passing None is what actually cuts it, and that is all this does.

WHY IT MATTERS ON H3. MiniMax H3 is a joint audio-video model: give it reference
audio and it drives mouth movement from that audio. That is exactly what you want on
a talking clip, and exactly what ruins one where the subject is silent over
background music — there the audio conditioning and the video conditioning pull in
opposite directions, and the motion that loses is the one from the reference video.

So it is a per-clip decision, not a per-workflow one, which is why it is a widget
rather than a wiring choice.

Returning None rather than silence is deliberate: silence is content, and a model
handed a silent track can faithfully reproduce silence where it would otherwise have
generated nothing.
"""


class AudioSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": True, "label_on": "audio connected", "label_off": "audio muted",
                    "tooltip": "Off passes nothing downstream — not silence, nothing.",
                }),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Leave connected; use the toggle to include or drop it."}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "switch"
    CATEGORY = "Ember/Utils"
    DESCRIPTION = "Include or drop an audio track without rewiring."

    def switch(self, enabled, audio=None):
        if not enabled:
            print("[Ember Audio Switch] muted — passing no audio downstream")
            return (None,)
        if audio is None:
            # Enabled but nothing wired. Say so: the failure downstream would be
            # "no audio" either way, and only one of those is a mistake.
            print("[Ember Audio Switch] enabled, but no audio is connected")
        return (audio,)


NODE_CLASS_MAPPINGS = {"EmberAudioSwitch": AudioSwitch}
NODE_DISPLAY_NAME_MAPPINGS = {"EmberAudioSwitch": "Ember Audio Switch"}
