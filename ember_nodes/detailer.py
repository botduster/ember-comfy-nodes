"""Ember Detailer: detect, enlarge, re-sample and paste back a region (eyes, face, skin).

Provenance, stated plainly. The orchestration below (FaceDetailer ->
DetailerForEach.do_detail -> enhance_detail) is derived from ComfyUI-Impact-Pack
by Dr.Lt.Data (GPL-3.0), version 8.28.3, and it calls that pack's internals at
run time. What this node changes relative to Impact's FaceDetailer:

  1. Paste-back resampler. After VAE decode, the refined crop is resized back to its
     source size with torch interpolation (`interpolation`, default bicubic with
     antialias), in float, instead of Impact's 8-bit PIL LANCZOS round trip.
  2. `color_match_strength` (0..1): per-channel mean/std of the refined crop are
     re-aligned to the source crop, cancelling VAE/sampler colour drift.
  3. `sharpness`: unsharp mask on luma only (BT.601, 5x5 Gaussian, sigma 1),
     added equally to R, G and B, so hue is kept and no colour fringing appears.
     Applied after the downscale, to the regenerated crop only.
  4. Optional `segs` input. When connected, detection is skipped and those SEGS are
     used as-is, so `bbox_detector` becomes optional. There is no `wildcard` widget.

Order of the paste-back stage: resize -> colour match -> sharpen -> paste with the
feathered mask.

Runtime dependencies (nothing is copied from ComfyUI-Impact-Subpack; detectors
arrive as inputs):
  impact.core            SEG, segs_scale_match, crop_condition_mask, make_sam_mask,
                         segs_bitwise_and_mask, segs_to_combined_mask, get_schedulers
  impact.utils           tensor_gaussian_blur_mask, tensor_resize, to_latent_image,
                         crop_ndarray4, to_tensor, tensor_paste, tensor_convert_rgba,
                         tensor_putalpha, tensor_get_size, tensor_convert_rgb, empty_pil_tensor
  impact.wildcards       process_wildcard_for_segs, process_with_loras
  impact.impact_sampling ksampler_wrapper
  ComfyUI core           nodes.InpaintModelConditioning / VAEDecodeTiled / ConditioningConcat,
                         comfy_extras.nodes_differential_diffusion, comfy.samplers
"""

import inspect
import logging
import time

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import comfy.samplers
import nodes

try:
    from comfy_extras import nodes_differential_diffusion
except Exception:
    nodes_differential_diffusion = None


def _impact():
    """Impact Pack modules, resolved at call time so custom-node load order never matters."""
    import impact.core as core
    import impact.utils as utils
    import impact.wildcards as wildcards
    from impact import impact_sampling
    return core, utils, wildcards, impact_sampling


def _differential_diffusion(model):
    return nodes_differential_diffusion.DifferentialDiffusion().execute(model)[0]


# ---------------------------------------------------------------------------
# Paste-back stage (the part that differs from Impact Pack)
# ---------------------------------------------------------------------------

INTERPOLATION_MODES = ["bicubic", "bilinear", "nearest-exact", "nearest", "area"]
_SMOOTH_MODES = {"bilinear", "bicubic"}  # the only modes that accept align_corners / antialias
_BT601 = (0.299, 0.587, 0.114)


def resize_back(image, width, height, mode="bicubic"):
    """BHWC float image -> width x height with torch interpolation, clamped to 0..1."""
    options = {"size": (height, width), "mode": mode}
    if mode in _SMOOTH_MODES:
        options["align_corners"] = False
        options["antialias"] = True
    out = F.interpolate(image.permute(0, 3, 1, 2), **options).clamp(0.0, 1.0)
    return out.permute(0, 2, 3, 1).contiguous()


def match_colour(refined, source, strength=1.0):
    """Align each channel's mean/std of `refined` to `source` (same size, BHWC), blended by strength."""
    if strength <= 0:
        return refined
    matched = refined.clone()
    for c in range(matched.shape[-1]):
        target = source[..., c]
        channel = matched[..., c]
        matched[..., c] = (channel - channel.mean()) / (channel.std() + 1e-6) * target.std() + target.mean()
    matched = matched.clamp(0.0, 1.0)
    if strength < 1.0:
        matched = (refined * (1.0 - strength) + matched * strength).clamp(0.0, 1.0)
    return matched


def sharpen_luma(image, amount, kernel_size=5, sigma=1.0):
    """Unsharp mask computed on BT.601 luma and added equally to R, G, B."""
    if amount <= 0:
        return image
    chw = image.permute(0, 3, 1, 2)
    luma = (chw * chw.new_tensor(_BT601).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    detail = luma - TF.gaussian_blur(luma, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    out = (chw + amount * detail).clamp(0.0, 1.0)
    return out.permute(0, 2, 3, 1).contiguous()


# ---------------------------------------------------------------------------
# Detail one crop (Impact core.enhance_detail + the paste-back stage)
# ---------------------------------------------------------------------------

def enhance_detail(image, model, clip, vae, guide_size, guide_size_for_bbox, max_size, bbox, seed, steps, cfg,
                   sampler_name, scheduler, positive, negative, denoise, noise_mask, force_inpaint,
                   wildcard_opt=None, wildcard_opt_concat_mode=None, detailer_hook=None,
                   refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None,
                   refiner_negative=None, control_net_wrapper=None, cycle=1,
                   inpaint_model=False, noise_mask_feather=0, scheduler_func=None,
                   vae_tiled_encode=False, vae_tiled_decode=False, interpolation="bicubic", sharpness=0.5,
                   color_match_strength=1.0):
    core, utils, wildcards, impact_sampling = _impact()

    if noise_mask is not None:
        noise_mask = utils.tensor_gaussian_blur_mask(noise_mask, noise_mask_feather)
        noise_mask = noise_mask.squeeze(3)
        if noise_mask_feather > 0 and "denoise_mask_function" not in model.model_options:
            model = _differential_diffusion(model)

    if wildcard_opt is not None and wildcard_opt != "":
        model, _, wildcard_positive = wildcards.process_with_loras(wildcard_opt, model, clip)
        if wildcard_opt_concat_mode == "concat":
            positive = nodes.ConditioningConcat().concat(positive, wildcard_positive)[0]
        else:
            positive = [wildcard_positive[0].copy()]
            if "pooled_output" in wildcard_positive[0][1]:
                positive[0][1]["pooled_output"] = wildcard_positive[0][1]["pooled_output"]
            elif "pooled_output" in positive[0][1]:
                del positive[0][1]["pooled_output"]

    h, w = image.shape[1], image.shape[2]
    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    if not force_inpaint and bbox_h >= guide_size and bbox_w >= guide_size:
        logging.info("Ember Detailer: segment skip (enough big)")
        return None, None

    upscale = guide_size / min(bbox_w, bbox_h) if guide_size_for_bbox else guide_size / min(w, h)
    new_w, new_h = int(w * upscale), int(h * upscale)

    if "aitemplate_keep_loaded" in model.model_options:
        max_size = min(4096, max_size)

    if new_w > max_size or new_h > max_size:
        upscale *= max_size / max(new_w, new_h)
        new_w, new_h = int(w * upscale), int(h * upscale)

    if not force_inpaint:
        if upscale <= 1.0:
            logging.info(f"Ember Detailer: segment skip [determined upscale factor={upscale}]")
            return None, None
        if new_w == 0 or new_h == 0:
            logging.info(f"Ember Detailer: segment skip [zero size={new_w, new_h}]")
            return None, None
    elif upscale <= 1.0 or new_w == 0 or new_h == 0:
        logging.info("Ember Detailer: force inpaint")
        upscale, new_w, new_h = 1.0, w, h

    if detailer_hook is not None:
        new_w, new_h = detailer_hook.touch_scaled_size(new_w, new_h)

    logging.info(f"Ember Detailer: segment upscale for ({bbox_w, bbox_h}) | crop region {w, h} x {upscale} -> {new_w, new_h}")
    upscaled_image = utils.tensor_resize(image, new_w, new_h)

    if detailer_hook is not None:
        upscaled_image = detailer_hook.post_upscale(upscaled_image, noise_mask)

    cnet_pils = None
    if control_net_wrapper is not None:
        positive, negative, cnet_pils = control_net_wrapper.apply(positive, negative, upscaled_image, noise_mask)
        model, cnet_pils2 = control_net_wrapper.doit_ipadapter(model)
        cnet_pils.extend(cnet_pils2)

    if detailer_hook is None or not detailer_hook.get_skip_sampling():
        if noise_mask is not None and inpaint_model:
            imc_encode = nodes.InpaintModelConditioning().encode
            if "noise_mask" in inspect.signature(imc_encode).parameters:
                positive, negative, latent_image = imc_encode(positive, negative, upscaled_image, vae, mask=noise_mask, noise_mask=True)
            else:
                logging.warning("[Ember Detailer] ComfyUI is an outdated version.")
                positive, negative, latent_image = imc_encode(positive, negative, upscaled_image, vae, noise_mask)
        else:
            latent_image = utils.to_latent_image(upscaled_image, vae, vae_tiled_encode=vae_tiled_encode)
            if noise_mask is not None:
                latent_image["noise_mask"] = noise_mask

        if detailer_hook is not None:
            latent_image = detailer_hook.post_encode(latent_image)

        refined_latent = latent_image
        sampler_opt = detailer_hook.get_custom_sampler() if detailer_hook is not None else None

        for i in range(cycle):
            if detailer_hook is not None:
                detailer_hook.set_steps((i, cycle))
                refined_latent = detailer_hook.cycle_latent(refined_latent)
                model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, _, denoise2 = \
                    detailer_hook.pre_ksample(model, seed + i, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
                noise, is_touched = detailer_hook.get_custom_noise(seed + i, torch.zeros(latent_image["samples"].size()), is_touched=False)
                if not is_touched:
                    noise = None
            else:
                model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, denoise2 = \
                    model, seed + i, steps, cfg, sampler_name, scheduler, positive, negative, denoise
                noise = None

            refined_latent = impact_sampling.ksampler_wrapper(
                model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2,
                refined_latent, denoise2, refiner_ratio, refiner_model, refiner_clip, refiner_positive, refiner_negative,
                noise=noise, scheduler_func=scheduler_func, sampler_opt=sampler_opt)

        if detailer_hook is not None:
            refined_latent = detailer_hook.pre_decode(refined_latent)

        start = time.time()
        if vae_tiled_decode:
            (refined_image,) = nodes.VAEDecodeTiled().decode(vae, refined_latent, 512)
            logging.info(f"[Ember Detailer] vae decoded (tiled) in {time.time() - start:.1f}s")
        else:
            try:
                refined_image = vae.decode(refined_latent["samples"])
            except Exception:
                logging.warning(f"[Ember Detailer] decode failed after {time.time() - start:.1f}s, retrying tiled 64...")
                refined_image = vae.decode_tiled(refined_latent["samples"], tile_x=64, tile_y=64)
            logging.info(f"[Ember Detailer] vae decoded in {time.time() - start:.1f}s")
    else:
        refined_image = upscaled_image

    if detailer_hook is not None:
        refined_image = detailer_hook.post_decode(refined_image)

    if len(refined_image.shape) == 5:  # video VAEs (e.g. WAN) decode a 1-frame batch as 5-D
        refined_image = refined_image.squeeze(0)

    refined_image = resize_back(refined_image, w, h, interpolation)
    refined_image = match_colour(refined_image, image, color_match_strength)
    refined_image = sharpen_luma(refined_image, sharpness)
    return refined_image.cpu(), cnet_pils


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

class EmberDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        core, _, _, _ = _impact()
        return {"required": {
                    "image": ("IMAGE",),
                    "model": ("MODEL", {"tooltip": "If `ImpactDummyInput` is connected here, sampling is skipped."}),
                    "clip": ("CLIP",),
                    "vae": ("VAE",),
                    "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                    "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                    "scheduler": (core.get_schedulers(),),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                    "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                    "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                    "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),

                    "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                    "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),

                    "sam_detection_hint": (["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area",
                                            "mask-points", "mask-point-bbox", "none"],),
                    "sam_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                    "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                    "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "sam_mask_hint_use_negative": (["False", "Small", "Outter"],),

                    "drop_size": ("INT", {"min": 1, "max": nodes.MAX_RESOLUTION, "step": 1, "default": 10}),
                    "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),

                    "interpolation": (INTERPOLATION_MODES, {
                        "default": "bicubic",
                        "tooltip": "Resampler for the paste-back resize after VAE decode (bicubic/bilinear get antialias)."}),
                    "sharpness": ("FLOAT", {
                        "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                        "tooltip": "Luma-only unsharp mask on the regenerated crop. 0 = off, 0.3 subtle, 1.0+ strong."}),
                    "color_match_strength": ("FLOAT", {
                        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                        "tooltip": "Re-align the regenerated crop's per-channel mean/std to the source. 0 = off, 1 = full."}),
                    },
                "optional": {
                    "bbox_detector": ("BBOX_DETECTOR", {
                        "tooltip": "Detector (faces/eyes). Optional when `segs` is connected."}),
                    "segs": ("SEGS", {
                        "tooltip": "Precomputed SEGS. When connected, bbox/SAM/segm detection is skipped entirely."}),
                    "sam_model_opt": ("SAM_MODEL",),
                    "segm_detector_opt": ("SEGM_DETECTOR",),
                    "detailer_hook": ("DETAILER_HOOK",),
                    "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                    "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    "tiled_encode": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "tiled_decode": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", "IMAGE")
    RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "mask", "detailer_pipe", "cnet_images")
    OUTPUT_IS_LIST = (False, True, True, False, False, True)
    FUNCTION = "doit"
    CATEGORY = "Ember/Detailer"
    DESCRIPTION = ("Detect (or take SEGS), enlarge, re-sample and paste back. Impact FaceDetailer behaviour with a "
                   "float paste-back resampler, colour match and luma sharpening.")

    @staticmethod
    def do_detail(image, segs, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seed, steps, cfg, sampler_name,
                  scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint, wildcard_opt=None,
                  detailer_hook=None, refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None,
                  refiner_negative=None, cycle=1, inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None,
                  tiled_encode=False, tiled_decode=False, interpolation="bicubic", sharpness=0.5, color_match_strength=1.0):
        core, utils, wildcards, _ = _impact()

        if len(image) > 1:
            raise Exception("[Ember Detailer] ERROR: image batches are not supported by do_detail.")

        image = image.clone()
        enhanced_alpha_list, enhanced_list, cropped_list, cnet_pil_list = [], [], [], []

        segs = core.segs_scale_match(segs, image.shape)
        new_segs = []

        wildcard_concat_mode = None
        if wildcard_opt is not None:
            if wildcard_opt.startswith("[CONCAT]"):
                wildcard_concat_mode = "concat"
                wildcard_opt = wildcard_opt[8:]
            wmode, wildcard_chooser = wildcards.process_wildcard_for_segs(wildcard_opt)
        else:
            wmode, wildcard_chooser = None, None

        def area(seg):
            return (seg.bbox[2] - seg.bbox[0]) * (seg.bbox[3] - seg.bbox[1])

        if wmode == "ASC":
            ordered_segs = sorted(segs[1], key=lambda s: (s.bbox[0], s.bbox[1]))
        elif wmode == "DSC":
            ordered_segs = sorted(segs[1], key=lambda s: (s.bbox[0], s.bbox[1]), reverse=True)
        elif wmode == "ASC-SIZE":
            ordered_segs = sorted(segs[1], key=area)
        elif wmode == "DSC-SIZE":
            ordered_segs = sorted(segs[1], key=area, reverse=True)
        else:
            ordered_segs = segs[1]

        if not (isinstance(model, str) and model == "DUMMY") and noise_mask_feather > 0 \
                and "denoise_mask_function" not in model.model_options:
            model = _differential_diffusion(model)

        for i, seg in enumerate(ordered_segs):
            # Crop from the working image, never seg.cropped_image, so overlapping segments see earlier pastes.
            cropped_image = utils.to_tensor(utils.crop_ndarray4(image.cpu().numpy(), seg.crop_region))
            mask = utils.tensor_gaussian_blur_mask(utils.to_tensor(seg.cropped_mask), feather)

            if (seg.cropped_mask == 0).all().item():
                logging.info("Ember Detailer: segment skip [empty mask]")
                continue

            cropped_mask = seg.cropped_mask if noise_mask else None

            if wildcard_chooser is not None and wmode != "LAB":
                seg_seed, wildcard_item = wildcard_chooser.get(seg)
            elif wildcard_chooser is not None and wmode == "LAB":
                seg_seed, wildcard_item = None, wildcard_chooser.get(seg)
            else:
                seg_seed, wildcard_item = None, None
            seg_seed = seed + i if seg_seed is None else seg_seed

            def crop_conditioning(conditioning):
                if isinstance(conditioning, str):  # placeholder conditioning (e.g. FLUX negative)
                    return conditioning
                return [[condition, {k: core.crop_condition_mask(v, image, seg.crop_region) if k == "mask" else v
                                     for k, v in details.items()}]
                        for condition, details in conditioning]

            cropped_positive = crop_conditioning(positive)
            cropped_negative = crop_conditioning(negative)

            if wildcard_item and wildcard_item.strip() == "[SKIP]":
                continue
            if wildcard_item and wildcard_item.strip() == "[STOP]":
                break

            orig_cropped_image = cropped_image.clone()
            if not (isinstance(model, str) and model == "DUMMY"):
                enhanced_image, cnet_pils = enhance_detail(
                    cropped_image, model, clip, vae, guide_size, guide_size_for_bbox, max_size,
                    seg.bbox, seg_seed, steps, cfg, sampler_name, scheduler,
                    cropped_positive, cropped_negative, denoise, cropped_mask, force_inpaint,
                    wildcard_opt=wildcard_item, wildcard_opt_concat_mode=wildcard_concat_mode,
                    detailer_hook=detailer_hook, refiner_ratio=refiner_ratio, refiner_model=refiner_model,
                    refiner_clip=refiner_clip, refiner_positive=refiner_positive, refiner_negative=refiner_negative,
                    control_net_wrapper=seg.control_net_wrapper, cycle=cycle, inpaint_model=inpaint_model,
                    noise_mask_feather=noise_mask_feather, scheduler_func=scheduler_func_opt,
                    vae_tiled_encode=tiled_encode, vae_tiled_decode=tiled_decode, interpolation=interpolation,
                    sharpness=sharpness, color_match_strength=color_match_strength)
            else:
                enhanced_image, cnet_pils = cropped_image, None

            if cnet_pils is not None:
                cnet_pil_list.extend(cnet_pils)

            if enhanced_image is not None:
                image = image.cpu()
                enhanced_image = enhanced_image.cpu()
                utils.tensor_paste(image, enhanced_image, (seg.crop_region[0], seg.crop_region[1]), mask)
                enhanced_list.append(enhanced_image)
                if detailer_hook is not None:
                    image = detailer_hook.post_paste(image)

                enhanced_image_alpha = utils.tensor_convert_rgba(enhanced_image)
                new_seg_image = enhanced_image.numpy()
                mask = utils.tensor_resize(mask, *utils.tensor_get_size(enhanced_image))
                utils.tensor_putalpha(enhanced_image_alpha, mask)
                enhanced_alpha_list.append(enhanced_image_alpha)
            else:
                new_seg_image = None

            cropped_list.append(orig_cropped_image)
            new_segs.append(core.SEG(new_seg_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox,
                                     seg.label, seg.control_net_wrapper))

        image_tensor = utils.tensor_convert_rgb(image)
        cropped_list.sort(key=lambda x: x.shape, reverse=True)
        enhanced_list.sort(key=lambda x: x.shape, reverse=True)
        enhanced_alpha_list.sort(key=lambda x: x.shape, reverse=True)
        return image_tensor, cropped_list, enhanced_list, enhanced_alpha_list, cnet_pil_list, (segs[0], new_segs)

    @staticmethod
    def enhance_face(image, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seed, steps, cfg, sampler_name,
                     scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint,
                     bbox_threshold, bbox_dilation, bbox_crop_factor,
                     sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                     sam_mask_hint_use_negative, drop_size,
                     bbox_detector=None, segm_detector=None, sam_model_opt=None, wildcard_opt=None, detailer_hook=None,
                     refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None,
                     cycle=1, inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None, tiled_encode=False,
                     tiled_decode=False, interpolation="bicubic", sharpness=0.5, color_match_strength=1.0, segs_opt=None):
        core, utils, _, _ = _impact()

        if segs_opt is not None:
            segs = core.segs_scale_match(segs_opt, image.shape)
        else:
            if bbox_detector is None:
                raise Exception("[Ember Detailer] ERROR: connect either `bbox_detector` or the optional `segs` input.")
            bbox_detector.setAux("face")  # default CLIPSeg prompt when a detector uses one
            segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size, detailer_hook=detailer_hook)
            bbox_detector.setAux(None)

            if sam_model_opt is not None:
                sam_mask = core.make_sam_mask(sam_model_opt, segs, image, sam_detection_hint, sam_dilation,
                                              sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                                              sam_mask_hint_use_negative)
                segs = core.segs_bitwise_and_mask(segs, sam_mask)
            elif segm_detector is not None:
                segm_segs = segm_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)
                if (hasattr(segm_detector, "override_bbox_by_segm") and segm_detector.override_bbox_by_segm and
                        not (detailer_hook is not None and not hasattr(detailer_hook, "override_bbox_by_segm"))):
                    segs = segm_segs
                else:
                    segs = core.segs_bitwise_and_mask(segs, core.segs_to_combined_mask(segm_segs))

        if len(segs[1]) > 0:
            enhanced_img, _, cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list, _ = EmberDetailer.do_detail(
                image, segs, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seed, steps, cfg,
                sampler_name, scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint,
                wildcard_opt, detailer_hook, refiner_ratio=refiner_ratio, refiner_model=refiner_model,
                refiner_clip=refiner_clip, refiner_positive=refiner_positive, refiner_negative=refiner_negative,
                cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather,
                scheduler_func_opt=scheduler_func_opt, tiled_encode=tiled_encode, tiled_decode=tiled_decode,
                interpolation=interpolation, sharpness=sharpness, color_match_strength=color_match_strength)
        else:
            enhanced_img, cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list = image, [], [], []

        mask = core.segs_to_combined_mask(segs)
        if len(cropped_enhanced) == 0:
            cropped_enhanced = [utils.empty_pil_tensor()]
        if len(cropped_enhanced_alpha) == 0:
            cropped_enhanced_alpha = [utils.empty_pil_tensor()]
        if len(cnet_pil_list) == 0:
            cnet_pil_list = [utils.empty_pil_tensor()]
        return enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list

    def doit(self, image, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
             positive, negative, denoise, feather, noise_mask, force_inpaint,
             bbox_threshold, bbox_dilation, bbox_crop_factor,
             sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
             sam_mask_hint_use_negative, drop_size, bbox_detector=None, wildcard="", cycle=1,
             interpolation="bicubic", sharpness=0.5, color_match_strength=1.0,
             sam_model_opt=None, segm_detector_opt=None, detailer_hook=None, inpaint_model=False, noise_mask_feather=0,
             scheduler_func_opt=None, tiled_encode=False, tiled_decode=False, segs=None):
        if not torch.is_tensor(image):
            raise Exception(
                "[Ember Detailer] `image` is not an image (got "
                f"{type(image).__name__}). This is almost always a stale node schema cached by the browser: "
                "hard-reload the page, then delete and re-add this node.")

        if len(image) > 1:
            logging.warning("[Ember Detailer] WARN: not designed for video detailing; use a batch detailer for video.")

        result_img, result_mask = None, None
        result_cropped_enhanced, result_cropped_enhanced_alpha, result_cnet_images = [], [], []
        for i, single_image in enumerate(image):
            enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list = EmberDetailer.enhance_face(
                single_image.unsqueeze(0), model, clip, vae, guide_size, guide_size_for, max_size, seed + i, steps, cfg,
                sampler_name, scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint,
                bbox_threshold, bbox_dilation, bbox_crop_factor,
                sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                sam_mask_hint_use_negative, drop_size, bbox_detector, segm_detector_opt, sam_model_opt, wildcard,
                detailer_hook, cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather,
                scheduler_func_opt=scheduler_func_opt, tiled_encode=tiled_encode, tiled_decode=tiled_decode,
                interpolation=interpolation, sharpness=sharpness, color_match_strength=color_match_strength,
                segs_opt=segs)

            result_img = enhanced_img if result_img is None else torch.cat((result_img, enhanced_img), dim=0)
            result_mask = mask if result_mask is None else torch.cat((result_mask, mask), dim=0)
            result_cropped_enhanced.extend(cropped_enhanced)
            result_cropped_enhanced_alpha.extend(cropped_enhanced_alpha)
            result_cnet_images.extend(cnet_pil_list)

        pipe = (model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt,
                detailer_hook, None, None, None, None)
        return result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask, pipe, result_cnet_images


NODE_CLASS_MAPPINGS = {"EmberDetailer": EmberDetailer}
NODE_DISPLAY_NAME_MAPPINGS = {"EmberDetailer": "Ember Detailer"}
