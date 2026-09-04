"""Pull one frame out of a video.

Two sources, because both are needed in practice:

  A FILE in ComfyUI's input folder — fine when you are working one clip at a time.
  CONNECTED FRAMES on `video_frames` — what lets a batch drive this without a human
  picking a filename for every clip. When connected it REPLACES the file entirely
  and nothing is read from disk.

`start_second` only means anything when the frame rate is known. From a file that
comes from the container; from connected frames it has to be supplied, so leaving
video_fps at 0 makes start_second inert rather than silently wrong.
"""

import os

import numpy as np
import torch

try:  # Available inside ComfyUI; absent when this module is imported for tests.
    import folder_paths
except ImportError:  # pragma: no cover
    folder_paths = None

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def list_video_files():
    if folder_paths is None:
        return []
    try:
        d = folder_paths.get_input_directory()
        return sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS)
    except Exception:
        return []


def sample_indices(total: int, start: int, count: int, interval: int):
    """Indices of the sampled frames, clamped to what the source actually has."""
    if total <= 0:
        return []
    step = max(1, int(interval))
    start = max(0, min(int(start), total - 1))
    return list(range(start, min(total, start + int(count) * step), step))


class VideoFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (list_video_files(), {"video_upload": True}),
                "start_second": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1,
                                           "tooltip": "Ignored unless the frame rate is known."}),
                "frame_count": ("INT", {"default": 10, "min": 1, "max": 9999, "step": 1,
                                        "tooltip": "How many frames are sampled, and so how far "
                                                   "selected_frame can reach."}),
                "frame_interval": ("INT", {"default": 1, "min": 1, "max": 300, "step": 1,
                                           "tooltip": "Take one frame every N. 1 samples every frame."}),
                "selected_frame": ("INT", {"default": 0, "min": 0, "max": 9998, "step": 1,
                                           "tooltip": "Which sampled frame to output. 0 with "
                                                      "start_second 0 is the very first frame."}),
            },
            "optional": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Connect a batch loader's frames. When connected this REPLACES the "
                               "file dropdown and nothing is read from disk.",
                }),
                "video_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Frame rate of the connected frames, so start_second can be turned "
                               "into an index. At 0, start_second is ignored.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "frame_index")
    FUNCTION = "extract"
    CATEGORY = "Ember/Video"
    DESCRIPTION = "Output a single frame from a video file or a connected frame batch."

    def extract(self, video, start_second, frame_count, frame_interval, selected_frame,
                video_frames=None, video_fps=0.0):
        if video_frames is not None:
            total = int(video_frames.shape[0])
            start = int(start_second * video_fps) if video_fps and video_fps > 0 else 0
            idx = sample_indices(total, start, frame_count, frame_interval)
            if not idx:
                raise RuntimeError("[Ember Video Frame] The connected batch is empty.")
            pick = idx[min(int(selected_frame), len(idx) - 1)]
            print("[Ember Video Frame] connected batch: %d frames, sampled %d, using index %d"
                  % (total, len(idx), pick))
            return (video_frames[pick:pick + 1], pick)

        return self._from_file(video, start_second, frame_count, frame_interval, selected_frame)

    def _from_file(self, video, start_second, frame_count, frame_interval, selected_frame):
        import av  # imported here so the module loads without PyAV for tests

        if folder_paths is None:
            raise RuntimeError("[Ember Video Frame] ComfyUI's folder_paths is unavailable.")
        path = os.path.join(folder_paths.get_input_directory(), video)
        if not os.path.exists(path):
            raise RuntimeError("[Ember Video Frame] Not found: %s" % path)

        with av.open(path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate or 24.0)
            start = int(max(0.0, float(start_second)) * fps)
            wanted = sample_indices(10 ** 9, start, frame_count, frame_interval)
            target = wanted[min(int(selected_frame), len(wanted) - 1)]

            # Decoded sequentially rather than seeking: seeking lands on the nearest
            # keyframe, so an exact index needs the frames counted anyway.
            for i, frame in enumerate(container.decode(stream)):
                if i == target:
                    arr = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
                    print("[Ember Video Frame] %s: frame %d at %.2f fps" % (video, target, fps))
                    return (torch.from_numpy(arr)[None, ...], target)

        raise RuntimeError(
            "[Ember Video Frame] %s has fewer than %d frames.\n"
            "-> Lower start_second, frame_count or selected_frame." % (video, target + 1)
        )


NODE_CLASS_MAPPINGS = {"EmberVideoFrame": VideoFrame}
NODE_DISPLAY_NAME_MAPPINGS = {"EmberVideoFrame": "Ember Video Frame"}
