"""Ember Save Image (No Metadata): ComfyUI's SaveImage, minus the embedded workflow.

Core SaveImage writes the prompt and the full workflow graph into every PNG. For
images that leave the pipeline (uploads, client delivery) that leaks the recipe and
bloats the file. This node writes pixels only:

- PNG: no text chunks. Only IHDR / IDAT / IEND, compress_level 4 (same as core).
- JPEG: no EXIF, ICC or comment segments. Pillow writes only the JFIF APP0 header,
  with optimize=True.

File naming and the returned `ui.images` records are identical to core SaveImage
(folder_paths.get_save_image_path), so anything that collects outputs from
history, such as our serverless handler, sees no difference.
"""

import logging
import os

import numpy as np
from PIL import Image

import folder_paths


class EmberSaveImageNoMetadata:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "Filename prefix. Supports ComfyUI tokens such as %date:yyyy-MM-dd%.",
                }),
                "format": (["PNG", "JPEG"], {
                    "default": "PNG",
                    "tooltip": "PNG is lossless. JPEG is smaller and lossy.",
                }),
                "quality": ("INT", {
                    "default": 95, "min": 1, "max": 100, "step": 1,
                    "tooltip": "JPEG quality 1-100. Ignored for PNG.",
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Ember/Image"
    DESCRIPTION = "Save images as PNG or JPEG with no workflow/prompt metadata embedded."

    def save_images(self, images, filename_prefix="ComfyUI", format="PNG", quality=95):
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir,
                                             images[0].shape[1], images[0].shape[0])

        results = []
        for index in range(images.shape[0]):
            pixels = (images[index].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            picture = Image.fromarray(pixels, mode="RGB")

            if format == "JPEG":
                file = f"{filename}_{counter:05}_.jpg"
                picture.save(os.path.join(full_output_folder, file), "JPEG",
                             quality=quality, optimize=True)
            else:
                file = f"{filename}_{counter:05}_.png"
                # No pnginfo argument, so no tEXt/iTXt chunks are written.
                picture.save(os.path.join(full_output_folder, file),
                             compress_level=self.compress_level)

            logging.info("[Ember SaveNoMeta] Saved %s/%s", subfolder or "output", file)
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {"EmberSaveImageNoMetadata": EmberSaveImageNoMetadata}
NODE_DISPLAY_NAME_MAPPINGS = {"EmberSaveImageNoMetadata": "Ember Save Image (No Metadata)"}
