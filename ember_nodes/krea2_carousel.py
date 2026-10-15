"""Ember Krea 2 carousel nodes: turn a list of poses into one Krea 2 Identity Edit instruction per slide.

Clean-room implementation, written 2026-10-15 from Ember's own spec (README "Krea 2 carousel nodes").
No third-party node code, prompt text or wording was used. Pure Python: no model, no network, no
torch at import time.

Ember Krea2 Pose List Prep (`EmberKrea2PoseListPrep`)
    Takes a pose list that Ember already wrote (the app asks its own vision model for poses, so no
    VLM runs on the GPU worker), accepts it as plain lines or JSON, and normalizes it: strips bullets
    and numbering, collapses whitespace, drops blanks and case-insensitive duplicates, caps each pose
    at MAX_POSE_CHARS and the list at MAX_SLIDES. `slide_count` 0 means "one slide per pose"; a
    positive count keeps the first N poses, and asking for more poses than exist is an error unless
    `if_too_few` = "use what is there".

Ember Krea2 Carousel Director (`EmberKrea2CarouselDirector`)
    Compiles each pose into an edit instruction for the Identity Edit LoRA:
        [trigger, ] keep-the-same-person sentence + continuity locks + camera rule
        + "New pose: <pose>." + optional shoot direction + optional extra text.
    (index/total are recorded in the manifest, not in the prompt: a slide number means nothing to the model.)
    `edit_prompts` and `negative_prompts` are LIST outputs (OUTPUT_IS_LIST), so ComfyUI runs the
    encode -> sample -> post chain once per slide and one job returns N images, all edited from the
    same master photo. Wire them to Krea2EditGroundedEncode.prompt (positive and negative).
    It also returns a JSON manifest (slides, settings the graph should use) and a plain-text report
    with warnings when a pose asks for something a lock forbids. Warnings never block the run.
"""

import json
import re

MAX_SLIDES = 8
MAX_POSE_CHARS = 300

CAMERA_MODES = ["auto", "locked", "free"]
CONTINUITY_LEVELS = ["balanced", "strong", "maximum"]
OUTPUT_MODES = ["all slides", "one slide"]
TOO_FEW_MODES = ["error", "use what is there"]

# What each continuity level holds fixed, in Ember's words. Each level includes the one before.
_LOCKS = {
    "balanced": ["face", "hair", "skin tone", "body shape", "outfit"],
    "strong": ["face", "hair", "skin tone", "body shape", "outfit", "accessories and jewellery", "makeup",
               "location", "lighting"],
    "maximum": ["face", "hair", "skin tone", "body shape", "outfit", "accessories and jewellery", "makeup",
                "location", "lighting", "background details", "colour grade", "photo style"],
}

_CAMERA_SENTENCES = {
    "auto": "Pick the camera angle and framing that suit the new pose, as a natural phone photo would.",
    "locked": "Keep the camera position, height, lens and framing exactly as in the reference photo.",
    "free": "The camera angle, distance and framing may change for this shot.",
}

# Recommended graph settings for the Identity Edit LoRA (public comfyui-krea2edit README: <= 2 MP,
# grounding 768, fit). The rest are Ember's carousel workflow values (Workflows/krea2_v3_carousel_api.json).
RECOMMENDED = {
    "identity_lora": "krea2_identity_edit_v1_2_r128.safetensors",
    "identity_lora_strength": 1.0,
    "grounding_px": 768,
    "fit_mode": "fit",
    "ref_boost": 4.0,
    "max_megapixels": 2.0,
    "steps": 10,
    "cfg": 1.0,
}

_BULLET = re.compile(r"^\s*(?:[-*•●▪>]+|\(?\d{1,3}[.):\]]|\(?[a-hA-H][.)\]](?=\s))\s*")
_SLIDE_LABEL = re.compile(r"^\s*(?:slide|pose|shot|photo)\s*#?\s*\d{1,3}\s*[:.)\-–—]\s*", re.I)

# (pattern, lock it conflicts with, message). Warnings only.
_CONFLICTS = [
    (re.compile(r"\b(?:change|swap|switch|different|new|another)\b[^.]{0,30}\b(?:outfit|clothes|clothing|dress|top)\b"
                r"|\b(?:outfit|clothes)\s+change\b|\bundress\w*|\btakes?\s+off\b", re.I),
     "outfit", "asks for a different outfit, but the outfit is locked"),
    (re.compile(r"\b(?:different|new|another)\s+(?:room|location|place|setting|background|scene)\b"
                r"|\b(?:moves?|walks?|goes)\s+(?:to|into|outside)\b", re.I),
     "location", "asks for a different location, but the location is locked"),
    (re.compile(r"\b(?:different|new|change\w*)\s+(?:hair|hairstyle|hair\s*colou?r)\b|\bhair\s+(?:up|down)\b", re.I),
     "hair", "asks for a hair change, but the hair is locked"),
    (re.compile(r"\b(?:sunset|night|golden hour|flash|neon|dark room)\b", re.I),
     "lighting", "describes different lighting, but the lighting is locked"),
]
_CAMERA_MOVES = re.compile(
    r"\b(?:selfie|mirror\s+selfie|close[- ]?up|from\s+(?:above|below|behind)|over\s+the\s+shoulder|"
    r"wide\s+shot|full[- ]body\s+shot|low\s+angle|high\s+angle|bird'?s[- ]eye|top[- ]down)\b", re.I)


# ---------------------------------------------------------------------------------------------------
# pose list parsing (shared by both nodes)
# ---------------------------------------------------------------------------------------------------

def _clean_pose(text):
    """Idempotent: prefixes are stripped until none is left, so cleaning an already-clean list (the
    Director re-reads what Pose List Prep wrote) can never change it."""
    text = re.sub(r"\s+", " ", str(text).replace("\u00a0", " ")).strip()
    while True:
        before = text
        text = _SLIDE_LABEL.sub("", text, count=1)
        text = _BULLET.sub("", text, count=1)
        text = text.strip().strip("\"'").strip().rstrip(" ,;")
        if text == before:
            break
    if len(text) > MAX_POSE_CHARS:
        cut = text[:MAX_POSE_CHARS]
        space = cut.rfind(" ")
        text = (cut[:space] if space >= MAX_POSE_CHARS // 2 else cut).rstrip(" ,;")
    return text


def _items_from_json(data):
    if isinstance(data, dict):
        for key in ("poses", "pose_list", "slides"):
            if key in data:
                return _items_from_json(data[key])
        raise ValueError("pose JSON object needs a 'poses' (or 'pose_list' / 'slides') array")
    if not isinstance(data, list):
        raise ValueError("pose JSON must be an array of strings or objects")
    items = []
    for entry in data:
        if isinstance(entry, str):
            items.append(entry)
        elif isinstance(entry, dict):
            for key in ("pose", "description", "text", "prompt"):
                if isinstance(entry.get(key), str):
                    items.append(entry[key])
                    break
            else:
                raise ValueError(f"pose object has no 'pose' / 'description' / 'text' / 'prompt' string: {entry!r}")
        elif entry is None:
            continue
        else:
            raise ValueError(f"unsupported pose entry type {type(entry).__name__}")
    return items


def parse_pose_list(text):
    """Return (poses, notes): the normalized, de-duplicated poses (uncapped) and what was changed."""
    notes = []
    raw = (text or "").strip()
    if not raw:
        return [], notes
    if raw[0] in "[{":
        try:
            items = _items_from_json(json.loads(raw))
            notes.append("parsed as JSON")
        except json.JSONDecodeError:
            items = raw.splitlines()
    else:
        items = raw.splitlines()
    poses, seen = [], set()
    for item in items:
        pose = _clean_pose(item)
        if not pose:
            continue
        key = pose.casefold()
        if key in seen:
            notes.append(f"dropped duplicate: {pose}")
            continue
        seen.add(key)
        poses.append(pose)
    return poses, notes


def select_poses(poses, slide_count, if_too_few="error"):
    """Apply the slide count and the MAX_SLIDES cap. Returns (poses, notes)."""
    notes = []
    if not poses:
        raise ValueError("the pose list is empty: give one pose per line (or a JSON array)")
    if slide_count < 0 or slide_count > MAX_SLIDES:
        raise ValueError(f"slide_count must be 0..{MAX_SLIDES}, got {slide_count}")
    want = slide_count or min(len(poses), MAX_SLIDES)
    if len(poses) < want:
        if if_too_few == "error":
            raise ValueError(f"slide_count is {want} but the pose list has only {len(poses)} pose(s)")
        notes.append(f"asked for {want} slides, only {len(poses)} pose(s): using {len(poses)}")
        want = len(poses)
    if len(poses) > want:
        notes.append(f"kept the first {want} of {len(poses)} poses")
    return poses[:want], notes


# ---------------------------------------------------------------------------------------------------
# instruction building
# ---------------------------------------------------------------------------------------------------

def _join_words(words):
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def build_instruction(pose, index, total, camera, continuity, trigger="", direction="", extra=""):
    if camera not in _CAMERA_SENTENCES:
        raise ValueError(f"unknown camera mode {camera!r}")
    if continuity not in _LOCKS:
        raise ValueError(f"unknown continuity level {continuity!r}")
    parts = []
    lead = "Show the same person from the reference photo in a new pose."
    parts.append(f"{trigger.strip()}, {lead}" if trigger and trigger.strip() else lead)
    parts.append(f"Keep her {_join_words(_LOCKS[continuity])} unchanged.")
    parts.append(_CAMERA_SENTENCES[camera])
    pose_text = pose.rstrip(".!? ")
    parts.append(f"New pose: {pose_text}.")
    if direction and direction.strip():
        parts.append(f"Shoot direction: {direction.strip().rstrip('.')}.")
    if extra and extra.strip():
        parts.append(extra.strip())
    return " ".join(parts)


def find_conflicts(pose, camera, continuity):
    locked = set(_LOCKS[continuity])
    out = []
    for pattern, lock, message in _CONFLICTS:
        if lock in locked and pattern.search(pose):
            out.append(message)
    if camera == "locked" and _CAMERA_MOVES.search(pose):
        out.append("asks for a camera move, but the camera is locked")
    return out


# ---------------------------------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------------------------------

class EmberKrea2PoseListPrep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_list": ("STRING", {"multiline": True, "default": "",
                                         "tooltip": "One pose per line, or a JSON array (strings or {\"pose\": ...} "
                                                    "objects, or {\"poses\": [...]}). Ember writes this."}),
                "slide_count": ("INT", {"default": 0, "min": 0, "max": MAX_SLIDES, "step": 1,
                                        "tooltip": "0 = one slide per pose (up to 8). N = the first N poses."}),
                "if_too_few": (TOO_FEW_MODES, {"default": "error",
                                               "tooltip": "What to do when slide_count is more than the poses given."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("pose_list", "slide_count", "report")
    FUNCTION = "prepare"
    CATEGORY = "Ember/Krea 2 Carousel"
    DESCRIPTION = "Validates and normalizes a pose list for the Ember Krea 2 Carousel Director."

    def prepare(self, pose_list, slide_count, if_too_few="error"):
        poses, notes = parse_pose_list(pose_list)
        poses, more = select_poses(poses, int(slide_count), if_too_few)
        notes += more
        report = f"{len(poses)} pose(s)" + ("".join(f"\n- {n}" for n in notes) if notes else "")
        return ("\n".join(poses), len(poses), report)


class EmberKrea2CarouselDirector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_list": ("STRING", {"multiline": True, "default": "",
                                         "tooltip": "One pose per line. Ignored when pose_list_in is connected."}),
                "slide_count": ("INT", {"default": 4, "min": 1, "max": MAX_SLIDES, "step": 1,
                                        "tooltip": "Ignored when slide_count_in is connected."}),
                "output_mode": (OUTPUT_MODES, {"default": "all slides",
                                               "tooltip": "one slide = only slide_index (re-roll one slide)."}),
                "slide_index": ("INT", {"default": 1, "min": 1, "max": MAX_SLIDES, "step": 1}),
                "camera": (CAMERA_MODES, {"default": "auto"}),
                "continuity": (CONTINUITY_LEVELS, {"default": "balanced",
                                                   "tooltip": "How much of the master photo is held fixed."}),
            },
            "optional": {
                "pose_list_in": ("STRING", {"forceInput": True}),
                "slide_count_in": ("INT", {"forceInput": True}),
                "master_image": ("IMAGE", {"tooltip": "Only read for its size (manifest) and to warn when missing."}),
                "character_trigger": ("STRING", {"default": "", "tooltip": "Creator trigger word, prefixed to every slide."}),
                "session_direction": ("STRING", {"multiline": True, "default": "",
                                                 "tooltip": "One line of direction shared by every slide."}),
                "extra_prompt": ("STRING", {"multiline": True, "default": "",
                                            "tooltip": "Appended to every slide as written."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "tooltip": "Base seed. slide_seeds = seed + slide number - 1, so each slide gets its own "
                                            "noise and a one-slide re-roll with the same seed reproduces that slide."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("edit_prompts", "negative_prompts", "selected_prompt", "manifest", "report",
                    "slide_count", "selected_index", "slide_seeds")
    OUTPUT_IS_LIST = (True, True, False, False, False, False, False, True)
    FUNCTION = "direct"
    CATEGORY = "Ember/Krea 2 Carousel"
    DESCRIPTION = "One Krea 2 Identity Edit instruction per pose (list output: one image per slide)."

    def direct(self, pose_list, slide_count, output_mode, slide_index, camera, continuity,
               pose_list_in=None, slide_count_in=None, master_image=None, character_trigger="",
               session_direction="", extra_prompt="", seed=0):
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"unknown output_mode {output_mode!r}")
        text = pose_list_in if pose_list_in is not None else pose_list
        count = int(slide_count_in) if slide_count_in is not None else int(slide_count)
        if count < 1:
            raise ValueError("slide_count must be at least 1")
        poses, notes = parse_pose_list(text)
        poses, more = select_poses(poses, count, "error")
        notes += more
        total = len(poses)

        slides, warnings = [], []
        for i, pose in enumerate(poses, 1):
            prompt = build_instruction(pose, i, total, camera, continuity, character_trigger,
                                       session_direction, extra_prompt)
            conflicts = find_conflicts(pose, camera, continuity)
            warnings += [f"slide {i} {c}" for c in conflicts]
            slides.append({"index": i, "pose": pose, "prompt": prompt, "warnings": conflicts,
                           "seed": (int(seed) + i - 1) % 0x10000000000000000})

        if not 1 <= slide_index <= total:
            if output_mode == "one slide":
                raise ValueError(f"slide_index {slide_index} is outside 1..{total}")
            selected = 1
        else:
            selected = slide_index
        if master_image is None:
            warnings.append("no master_image connected: the manifest has no source size")

        chosen = slides if output_mode == "all slides" else [slides[selected - 1]]
        prompts = [s["prompt"] for s in chosen]
        seeds = [s["seed"] for s in chosen]
        manifest = {
            "version": 1,
            "node": "EmberKrea2CarouselDirector",
            "slide_count": total,
            "output_mode": output_mode,
            "selected_index": selected,
            "camera": camera,
            "continuity": continuity,
            "locks": _LOCKS[continuity],
            "character_trigger": character_trigger.strip(),
            "source_size": None if master_image is None else
            {"width": int(master_image.shape[2]), "height": int(master_image.shape[1])},
            "recommended": dict(RECOMMENDED),
            "slides": slides,
            "notes": notes,
            "warnings": warnings,
        }
        lines = [f"{total} slide(s), camera {camera}, continuity {continuity}, output {output_mode}"]
        lines += [f"- {n}" for n in notes]
        lines += [f"WARNING: {w}" for w in warnings]
        return (prompts, [""] * len(prompts), slides[selected - 1]["prompt"],
                json.dumps(manifest, ensure_ascii=False, indent=2), "\n".join(lines), total, selected, seeds)


NODE_CLASS_MAPPINGS = {
    "EmberKrea2PoseListPrep": EmberKrea2PoseListPrep,
    "EmberKrea2CarouselDirector": EmberKrea2CarouselDirector,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EmberKrea2PoseListPrep": "Ember Krea2 Pose List Prep",
    "EmberKrea2CarouselDirector": "Ember Krea2 Carousel Director",
}
