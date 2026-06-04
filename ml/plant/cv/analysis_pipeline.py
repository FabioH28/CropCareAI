"""End-to-end plant computer vision analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ml.common.metrics import agreement_ratio, mask_coverage, severity_from_ratio, urgency_from_status
from ml.common.preprocessing import (
    contrast_stretch,
    gamma_correction,
    histogram_equalize,
    load_rgb_image,
    median_denoise,
    resize_image,
    rgb_to_hsv_array,
    to_grayscale,
    unsharp_mask,
    wiener_restore,
)
from ml.common.transforms import compute_excess_green, fft_magnitude_spectrum, frequency_filter, haar_wavelet_decompose
from ml.common.utils import ensure_dir, json_ready, min_max_normalize, safe_divide, slugify
from ml.common.visualization import heatmap, overlay_boundaries, overlay_mask, save_array_as_image, stack_images_grid
from ml.plant.cv.features import extract_feature_bundle
from ml.plant.cv.sam_leaf import sam_leaf_mask
from ml.plant.cv.segmentation import (
    build_leaf_mask,
    clustering_segmentation,
    extract_leaf_roi,
    graph_cut_segmentation,
    lesion_score_map,
    region_growing_segmentation,
    simple_superpixels,
    split_and_merge_segmentation,
    threshold_segmentation,
)

# Keep the "core vs supplementary" split near the top so the rest of the file
# reads like one pipeline instead of a loose bag of CV methods.
CORE_TOPICS = [
    "digital_image_processing_basics",
    "basic_image_manipulations",
    "spatial_and_resolution_changes",
    "intensity_transformations_and_spatial_filtering",
    "image_restoration_and_reconstruction",
    "color_image_processing",
    "segmentation_for_leaf_and_lesion_isolation",
    "feature_extraction_and_pattern_classification",
    "cv_plus_ai_smart_farming_integration",
    "full_cv_pipeline_demo",
]

SUPPLEMENTARY_TOPICS = [
    "frequency_domain_analysis_for_interpretability",
    "wavelet_analysis_for_texture_comparison",
    "region_growing_segmentation_comparison",
    "split_merge_segmentation_comparison",
    "superpixel_segmentation_comparison",
]

CORE_SEGMENTATION_METHODS = ("thresholding", "clustering", "graph_cut")
COMPARISON_ONLY_SEGMENTATION_METHODS = ("region_growing", "split_merge", "superpixels")


# Small helpers for exported images and filenames.
def _wavelet_panel(wavelet_data: dict[str, np.ndarray]) -> np.ndarray:
    tiles = [
        ("approximation", wavelet_data["approximation"]),
        ("horizontal", np.abs(wavelet_data["horizontal"])),
        ("vertical", np.abs(wavelet_data["vertical"])),
        ("diagonal", np.abs(wavelet_data["diagonal"])),
    ]
    return stack_images_grid(tiles, columns=2)


def _register_output(
    outputs: list[dict[str, str]],
    output_dir: Path,
    stage: str,
    label: str,
    array: np.ndarray,
) -> dict[str, str]:
    filename = f"{len(outputs) + 1:02d}_{slugify(stage)}_{slugify(label)}.png"
    relative_path = Path(filename)
    save_array_as_image(array, output_dir / relative_path)
    entry = {
        "stage": stage,
        "label": label,
        "file_name": filename,
        "relative_path": relative_path.as_posix(),
    }
    outputs.append(entry)
    return entry


def _prepare_localization_image(original_image: Image.Image, target_side: int = 384) -> np.ndarray:
    resized = original_image.resize((target_side, target_side), Image.Resampling.BILINEAR)
    return np.asarray(resized.convert("RGB"), dtype=np.uint8)


def _resize_array(array: np.ndarray, size: tuple[int, int], method: str = "bilinear") -> np.ndarray:
    if method == "nearest":
        resample = Image.Resampling.NEAREST
    elif method == "bicubic":
        resample = Image.Resampling.BICUBIC
    else:
        resample = Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(array.astype(np.uint8)).resize(size, resample), dtype=np.uint8)


def _extract_lesion_regions(
    lesion_mask: np.ndarray,
    leaf_mask: np.ndarray,
    *,
    min_component_ratio: float = 0.001,
) -> list[dict[str, Any]]:
    supported_mask = np.asarray(lesion_mask, dtype=bool) & np.asarray(leaf_mask, dtype=bool)
    labeled_mask, component_count = ndimage.label(supported_mask)
    leaf_pixels = max(int(np.asarray(leaf_mask, dtype=bool).sum()), 1)
    minimum_pixels = max(8, int(round(leaf_pixels * min_component_ratio)))

    raw_regions: list[dict[str, Any]] = []
    for component_index in range(1, component_count + 1):
        component_mask = labeled_mask == component_index
        area_pixels = int(component_mask.sum())
        if area_pixels <= 0:
            continue

        ys, xs = np.where(component_mask)
        if ys.size == 0 or xs.size == 0:
            continue

        raw_regions.append(
            {
                "component_index": component_index,
                "area_pixels": area_pixels,
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                "centroid_xy": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
                "area_percentage_of_leaf": round((area_pixels / leaf_pixels) * 100.0, 2),
            }
        )

    if not raw_regions:
        return []

    filtered_regions = [region for region in raw_regions if region["area_pixels"] >= minimum_pixels]
    if not filtered_regions:
        filtered_regions = [max(raw_regions, key=lambda region: region["area_pixels"])]

    filtered_regions.sort(key=lambda region: region["area_pixels"], reverse=True)
    for display_index, region in enumerate(filtered_regions, start=1):
        region["display_index"] = display_index
    return filtered_regions


def _annotate_lesion_regions(
    image_array: np.ndarray,
    lesion_mask: np.ndarray,
    lesion_regions: list[dict[str, Any]],
) -> np.ndarray:
    base_image = overlay_mask(image_array, lesion_mask, color=(255, 70, 70), alpha=0.28)
    pil_image = Image.fromarray(base_image)
    draw = ImageDraw.Draw(pil_image)

    for region in lesion_regions:
        left, top, right, bottom = region["bbox_xyxy"]
        centroid_x, centroid_y = region["centroid_xy"]
        label_text = str(region["display_index"])

        draw.rectangle((left, top, right - 1, bottom - 1), outline=(255, 235, 80), width=2)
        draw.ellipse((centroid_x - 3, centroid_y - 3, centroid_x + 3, centroid_y + 3), fill=(255, 235, 80))

        text_left = left + 4
        text_top = max(2, top - 14)
        draw.rectangle((text_left - 2, text_top - 1, text_left + 10, text_top + 10), fill=(40, 20, 20))
        draw.text((text_left, text_top), label_text, fill=(255, 235, 80))

    return np.asarray(pil_image, dtype=np.uint8)


# These lightweight heuristics keep the raw CV preview readable even before the
# deep model has the final say.
def _infer_crop(filename: str, crop_hint: str | None) -> str:
    if crop_hint:
        return crop_hint.lower()
    lower_name = filename.lower()
    for candidate in ("tomato", "potato", "pepper", "corn", "grape", "apple"):
        if candidate in lower_name:
            return candidate
    return "unknown_leaf_crop"


def _predict_disease(features: dict[str, dict[str, float]], health_status: str) -> str | None:
    if health_status == "healthy":
        return None

    infected_area = features["indices"]["infected_area_percentage"]
    lesion_brownness = features["indices"]["lesion_brownness"]
    high_frequency_ratio = features["frequency"]["high_to_low_ratio"]
    entropy = features["lesion_texture"]["entropy"]

    if infected_area > 18.0 and lesion_brownness > 0.08 and entropy > 2.0:
        return "early_blight_like_pattern"
    if infected_area > 12.0 and high_frequency_ratio > 0.65:
        return "late_blight_like_pattern"
    if lesion_brownness < 0.02:
        return "chlorosis_or_nutrient_stress_pattern"
    return "leaf_spot_like_pattern"


def _classify_baseline(leaf_mask: np.ndarray, threshold_mask: np.ndarray) -> dict[str, Any]:
    infected_ratio = mask_coverage(threshold_mask, leaf_mask)
    if infected_ratio < 0.04:
        health_status = "healthy"
    elif infected_ratio < 0.10:
        health_status = "suspicious"
    else:
        health_status = "diseased"

    severity_level = severity_from_ratio(infected_ratio)
    urgency_level = urgency_from_status(health_status, severity_level)
    confidence = min(0.94, 0.55 + infected_ratio * 1.8)

    return {
        "method": "baseline_thresholding",
        "health_status": health_status,
        "confidence_score": round(confidence, 3),
        "severity_level": severity_level,
        "urgency_level": urgency_level,
        "infected_area_percentage": round(infected_ratio * 100.0, 2),
    }


def _classify_improved(
    leaf_mask: np.ndarray,
    consensus_mask: np.ndarray,
    method_masks: dict[str, np.ndarray],
    features: dict[str, dict[str, float]],
) -> dict[str, Any]:
    infected_ratio = mask_coverage(consensus_mask, leaf_mask)
    agreement = agreement_ratio(list(method_masks.values()))
    lesion_brownness = max(features["indices"]["lesion_brownness"], 0.0)
    texture_entropy = features["lesion_texture"]["entropy"]
    high_frequency_ratio = features["frequency"]["high_to_low_ratio"]

    disease_score = (
        0.42 * infected_ratio
        + 0.18 * min(1.0, lesion_brownness * 6.0)
        + 0.15 * min(1.0, texture_entropy / 4.0)
        + 0.10 * min(1.0, high_frequency_ratio)
        + 0.15 * agreement
    )

    if disease_score < 0.19:
        health_status = "healthy"
    elif disease_score < 0.36:
        health_status = "suspicious"
    else:
        health_status = "diseased"

    severity_level = severity_from_ratio(infected_ratio)
    urgency_level = urgency_from_status(health_status, severity_level)
    confidence = np.clip(0.58 + disease_score * 0.55, 0.55, 0.98)

    return {
        "method": "consensus_multistage_cv",
        "health_status": health_status,
        "confidence_score": round(float(confidence), 3),
        "severity_level": severity_level,
        "urgency_level": urgency_level,
        "infected_area_percentage": round(infected_ratio * 100.0, 2),
        "disease_score": round(float(disease_score), 3),
        "segmentation_agreement": round(float(agreement), 3),
    }


def _isolate_leaf_mask(image_array: np.ndarray) -> np.ndarray:
    """SAM-based leaf isolation with a classical fallback.

    SAM segments the leaf by object shape, so it separates a green leaf from a
    same-coloured (green-on-green) background that defeats the colour-based
    ``build_leaf_mask``. If SAM is unavailable (no weights / torch, or disabled
    via CROPCARE_USE_SAM=0) or returns an implausible mask, we fall back to the
    hand-built classical mask. Every lesion mask is intersected with this leaf
    mask, so a clean leaf mask is what keeps lesions on the leaf and the
    infected-area % honest.
    """
    sam_mask = sam_leaf_mask(image_array)
    if sam_mask is not None:
        fraction = float(sam_mask.mean())
        if 0.04 <= fraction <= 0.97:
            return ndimage.binary_fill_holes(sam_mask)
    return build_leaf_mask(image_array)


# Main flow: isolate the leaf first, run the practical CV stack, then package
# the outputs in one backend-friendly JSON payload.
def analyze_plant_image(image_path: str | Path, output_dir: str | Path, crop_hint: str | None = None) -> dict[str, Any]:
    image_path = Path(image_path).resolve()
    output_dir = ensure_dir(output_dir)

    original_image = load_rgb_image(image_path)
    original_full_array = np.asarray(original_image.convert("RGB"), dtype=np.uint8)

    # First pass: find the leaf in the wider scene so the later steps spend less
    # time reacting to soil, sky, hands, or other field clutter.
    scene_resized_image = resize_image(original_image, (256, 256), method="bilinear")
    scene_resized_array = np.asarray(scene_resized_image, dtype=np.uint8)
    localization_array = _prepare_localization_image(original_image, target_side=384)
    localization_mask = build_leaf_mask(localization_array)
    localization_roi = extract_leaf_roi(localization_array, localization_mask, padding_ratio=0.14)

    if localization_roi["leaf_detected"]:
        left_n, top_n, right_n, bottom_n = localization_roi["bbox_normalized_xyxy"]
        original_width, original_height = original_image.size
        left = int(round(left_n * original_width))
        top = int(round(top_n * original_height))
        right = int(round(right_n * original_width))
        bottom = int(round(bottom_n * original_height))
        right = max(right, left + 1)
        bottom = max(bottom, top + 1)
        full_context_crop = original_full_array[top:bottom, left:right].copy()
        full_leaf_mask = _isolate_leaf_mask(full_context_crop)
        roi = extract_leaf_roi(full_context_crop, full_leaf_mask, padding_ratio=0.06)
        context_source = roi["context_crop"]
        isolated_source = roi["isolated_crop"]
        leaf_detected = bool(roi["leaf_detected"])
        roi_bbox_xyxy = [left, top, right, bottom]
        roi_bbox_normalized_xyxy = [left_n, top_n, right_n, bottom_n]
        roi_fill_ratio = float(roi["roi_fill_ratio"])
        leaf_area_ratio = float(roi["leaf_area_ratio"])
    else:
        context_source = original_full_array.copy()
        isolated_source = np.full_like(original_full_array, 255, dtype=np.uint8)
        leaf_detected = False
        roi_bbox_xyxy = [0, 0, original_image.size[0], original_image.size[1]]
        roi_bbox_normalized_xyxy = [0.0, 0.0, 1.0, 1.0]
        roi_fill_ratio = 0.0
        leaf_area_ratio = 0.0

    # Build the standard analysis views used by both the CV measurements and the
    # image set we save back to the app.
    primary_dl_source = isolated_source if leaf_detected else scene_resized_array
    context_resized = _resize_array(context_source, (256, 256), method="bilinear")
    primary_dl_resized = _resize_array(primary_dl_source, (256, 256), method="bilinear")
    resized_image = Image.fromarray(context_resized)
    grayscale_image = to_grayscale(resized_image)

    original_array = context_resized
    grayscale_array = np.asarray(grayscale_image, dtype=np.uint8)

    nearest_resized = np.asarray(resize_image(original_image, (256, 256), method="nearest"), dtype=np.uint8)
    bicubic_resized = np.asarray(resize_image(original_image, (256, 256), method="bicubic"), dtype=np.uint8)

    # Gentle enhancement first, then restoration, so lesion detail is easier to
    # segment without making the image look artificial.
    contrast_image = contrast_stretch(original_array)
    gamma_image = gamma_correction(contrast_image, gamma=0.9)
    enhanced_image = unsharp_mask(gamma_image, sigma=1.0, amount=1.15)

    median_restored = median_denoise(enhanced_image, size=3)
    restored_image = wiener_restore(median_restored, size=5)

    equalized_gray = histogram_equalize(grayscale_array)
    _, fft_spectrum = fft_magnitude_spectrum(equalized_gray)
    frequency_data = frequency_filter(equalized_gray, radius_ratio=0.10)
    wavelet_data = haar_wavelet_decompose(equalized_gray)

    # Thresholding, clustering, and graph cut drive the deployed lesion mask.
    # The other methods stay here as comparison views for demos and reports.
    leaf_mask = _isolate_leaf_mask(restored_image)
    lesion_score = lesion_score_map(restored_image, leaf_mask)
    exg = compute_excess_green(restored_image)

    threshold_mask = threshold_segmentation(restored_image, leaf_mask)
    region_mask = region_growing_segmentation(restored_image, leaf_mask)
    split_merge_mask = split_and_merge_segmentation(restored_image, leaf_mask)
    clustering_mask = clustering_segmentation(restored_image, leaf_mask)
    superpixel_labels, superpixel_mask = simple_superpixels(restored_image, leaf_mask)
    graph_cut_mask = graph_cut_segmentation(restored_image, leaf_mask)

    all_method_masks = {
        "thresholding": threshold_mask,
        "region_growing": region_mask,
        "split_merge": split_merge_mask,
        "clustering": clustering_mask,
        "superpixels": superpixel_mask,
        "graph_cut": graph_cut_mask,
    }
    core_method_masks = {name: all_method_masks[name] for name in CORE_SEGMENTATION_METHODS}
    # Vote with a small spatial tolerance. A spot's core (caught by thresholding)
    # and its surrounding halo (caught by clustering) are adjacent but rarely the
    # SAME pixels, so a strict per-pixel 2-of-3 vote drops many real lesions on
    # multi-spot diseases (e.g. rust). Dilating each method mask slightly before
    # voting lets the methods agree on a lesion even when one caught its core and
    # another its rim — a large recall gain, still a 2-of-3 consensus, and kept on
    # the leaf by the (SAM-isolated) leaf mask.
    vote_structure = np.ones((3, 3), dtype=bool)
    stacked_votes = np.stack(
        [ndimage.binary_dilation(mask, structure=vote_structure).astype(np.uint8)
         for mask in core_method_masks.values()],
        axis=0,
    )
    consensus_vote = (stacked_votes.sum(axis=0) >= 2) & leaf_mask

    # Absolute healthy-tissue gate. Per-image lesion scoring stretches contrast so
    # even a healthy leaf shows "suspicious" pixels (a documented normalization
    # artifact). We anchor to the leaf's OWN healthy colour — the mean RGB of its
    # greenest pixels — and keep a consensus pixel only when it is genuinely far
    # from that healthy tone. Healthy leaves collapse toward 0% infected; real
    # lesions (brown / orange / yellow rust, all far from green) survive. Being
    # relative to the leaf's own green, it adapts to any crop and lighting.
    rgb_norm = np.asarray(restored_image, dtype=np.float32) / 255.0
    if np.any(leaf_mask):
        # Compare in chromaticity (intensity-normalised colour) so within-leaf
        # brightness changes — glossy highlights, shadow, a pale midrib — do NOT
        # look like lesions; only a genuine HUE shift (green -> brown/orange/
        # yellow) does. This is what stops healthy leaves reading as diseased.
        chroma = rgb_norm / (rgb_norm.sum(axis=-1, keepdims=True) + 1e-6)
        green_cutoff = float(np.percentile(exg[leaf_mask], 60))
        healthy_sel = leaf_mask & (exg >= green_cutoff)
        if not np.any(healthy_sel):
            healthy_sel = leaf_mask
        healthy_chroma = chroma[healthy_sel].reshape(-1, 3).mean(axis=0)
        chroma_distance = np.linalg.norm(chroma - healthy_chroma, axis=-1)
        # A lesion is either (a) clearly off-colour from the healthy tone, OR
        # (b) warm-hued — red/orange/yellow — which is what rust/blight pustules
        # are. The warm-hue test is illumination-invariant and catches subtle
        # yellow spots that sit close to green in plain chroma distance, while a
        # green shadow or a darker-but-still-green patch stays rejected (it is
        # neither far in chroma nor warm). This breaks the recall/precision knot
        # that a single chroma threshold could not.
        hsv = rgb_to_hsv_array(rgb_norm)
        warm_hue = ((hsv[..., 0] < 0.18) | (hsv[..., 0] > 0.95)) & (hsv[..., 1] > 0.22)
        lesion_gate = (chroma_distance > 0.18) | warm_hue

        # Absolute colour-anomaly detector. The three voting methods rely on a
        # per-image-normalised lesion score + a morphological opening, which
        # together erase dense/small spots (many pustules normalise to mid-range
        # and the opening removes the little blobs) — so on a heavily, finely
        # spotted leaf the 2-of-3 vote finds almost nothing. Independently of the
        # vote, a pixel that is warm (orange/brown/red), saturated AND clearly
        # off-green IS a lesion regardless of how the rest of the image normalises.
        # Unioning this in recovers fine/dense spotting while staying near 0% on
        # healthy green tissue (healthy green is neither warm nor off-green).
        warm_strong = ((hsv[..., 0] < 0.15) | (hsv[..., 0] > 0.96)) & (hsv[..., 1] > 0.32)
        absolute_lesion = warm_strong & (exg < 0.06) & leaf_mask
        absolute_lesion = ndimage.binary_opening(absolute_lesion, np.ones((2, 2), dtype=bool))
        # Two decoupled channels (so detection sensitivity does not drive the
        # diagnosis):
        #   HIGHLIGHT (sensitive) — every colour-anomalous spot, for the marked-
        #   spots overlay and the displayed affected-area %.
        #   DECISION (strict) — only the confident 2-of-3 consensus, used for the
        #   CV health VERDICT + severity, so healthy field leaves with blemishes
        #   are not called diseased. In the deployed app the DL cascade is the
        #   disease authority; this strict CV verdict is just the fallback.
        consensus_mask = (consensus_vote & lesion_gate) | absolute_lesion
        decision_mask = consensus_vote & lesion_gate
    else:
        consensus_mask = np.zeros_like(leaf_mask)
        decision_mask = np.zeros_like(leaf_mask)

    # Once the lesion mask is stable enough, extract measurements the backend can
    # explain: coverage, texture, brownness, and severity clues.
    features = extract_feature_bundle(
        image_array=restored_image,
        leaf_mask=leaf_mask,
        lesion_mask=consensus_mask,
        gray_array=equalized_gray,
        frequency_data=frequency_data,
        wavelet_data=wavelet_data,
        segmentation_masks=core_method_masks,
    )

    baseline = _classify_baseline(leaf_mask, threshold_mask)
    # Verdict + severity from the strict DECISION mask; displayed affected-area %
    # from the sensitive HIGHLIGHT mask so it matches the spots shown to the user.
    improved = _classify_improved(leaf_mask, decision_mask, core_method_masks, features)
    improved["decision_area_percentage"] = improved["infected_area_percentage"]
    improved["infected_area_percentage"] = round(mask_coverage(consensus_mask, leaf_mask) * 100.0, 2)
    improved["predicted_disease"] = _predict_disease(features, improved["health_status"])
    improved["predicted_crop"] = _infer_crop(image_path.name, crop_hint)
    lesion_regions = _extract_lesion_regions(consensus_mask, leaf_mask)
    annotated_lesions = _annotate_lesion_regions(restored_image, consensus_mask, lesion_regions)
    largest_region_percentage = lesion_regions[0]["area_percentage_of_leaf"] if lesion_regions else 0.0
    mean_region_percentage = (
        round(float(np.mean([region["area_percentage_of_leaf"] for region in lesion_regions])), 2)
        if lesion_regions
        else 0.0
    )

    # Save the visual trail so the app and report can show what each major stage did.
    outputs: list[dict[str, str]] = []
    scene_output = _register_output(outputs, output_dir, "basics", "original_scene_resized", scene_resized_array)
    roi_output = _register_output(outputs, output_dir, "roi", "leaf_context_crop", context_resized)
    roi_primary_output = _register_output(outputs, output_dir, "roi", "leaf_isolated_primary_input", primary_dl_resized)
    _register_output(outputs, output_dir, "basics", "analysis_source_rgb", original_array)
    _register_output(outputs, output_dir, "basics", "grayscale", grayscale_array)
    _register_output(outputs, output_dir, "manipulation", "resize_nearest", nearest_resized)
    _register_output(outputs, output_dir, "manipulation", "resize_bicubic", bicubic_resized)
    _register_output(outputs, output_dir, "enhancement", "contrast_unsharp", enhanced_image)
    _register_output(outputs, output_dir, "restoration", "median_wiener_restored", restored_image)
    _register_output(outputs, output_dir, "color_analysis", "excess_green_heatmap", heatmap(exg))
    _register_output(outputs, output_dir, "color_analysis", "lesion_score_heatmap", heatmap(lesion_score))
    _register_output(outputs, output_dir, "frequency", "fft_magnitude_spectrum", fft_spectrum)
    _register_output(outputs, output_dir, "frequency", "low_pass_reconstruction", frequency_data["low_pass"])
    _register_output(outputs, output_dir, "frequency", "high_pass_reconstruction", np.abs(frequency_data["high_pass"]))
    _register_output(outputs, output_dir, "wavelets", "haar_wavelet_panel", _wavelet_panel(wavelet_data))
    _register_output(outputs, output_dir, "roi", "localization_leaf_mask_overlay", overlay_mask(localization_array, localization_mask, color=(80, 220, 120)))
    _register_output(outputs, output_dir, "segmentation", "leaf_mask_overlay", overlay_mask(restored_image, leaf_mask, color=(80, 220, 120)))
    _register_output(outputs, output_dir, "segmentation", "threshold_overlay", overlay_mask(restored_image, threshold_mask))
    _register_output(outputs, output_dir, "segmentation", "region_growing_overlay", overlay_mask(restored_image, region_mask, color=(255, 180, 0)))
    _register_output(outputs, output_dir, "segmentation", "split_merge_overlay", overlay_mask(restored_image, split_merge_mask, color=(160, 120, 255)))
    _register_output(outputs, output_dir, "segmentation", "clustering_overlay", overlay_mask(restored_image, clustering_mask, color=(0, 210, 255)))
    _register_output(outputs, output_dir, "segmentation", "superpixels_overlay", overlay_boundaries(overlay_mask(restored_image, superpixel_mask, color=(255, 80, 160)), superpixel_labels))
    _register_output(outputs, output_dir, "segmentation", "graph_cut_overlay", overlay_mask(restored_image, graph_cut_mask, color=(255, 60, 60)))
    _register_output(outputs, output_dir, "segmentation", "consensus_overlay", overlay_mask(restored_image, consensus_mask, color=(255, 0, 0)))
    _register_output(outputs, output_dir, "postprocessing", "marked_disease_spots", annotated_lesions)
    _register_output(
        outputs,
        output_dir,
        "comparison",
        "core_segmentation_comparison_grid",
        stack_images_grid(
            [
                ("threshold", overlay_mask(restored_image, threshold_mask)),
                ("clustering", overlay_mask(restored_image, clustering_mask, color=(0, 210, 255))),
                ("graph_cut", overlay_mask(restored_image, graph_cut_mask, color=(255, 60, 60))),
                ("consensus", overlay_mask(restored_image, consensus_mask, color=(255, 0, 0))),
            ],
            columns=2,
        ),
    )
    _register_output(
        outputs,
        output_dir,
        "comparison",
        "supplementary_segmentation_comparison_grid",
        stack_images_grid(
            [
                ("region", overlay_mask(restored_image, region_mask, color=(255, 180, 0))),
                ("split_merge", overlay_mask(restored_image, split_merge_mask, color=(160, 120, 255))),
                ("superpixels", overlay_boundaries(overlay_mask(restored_image, superpixel_mask, color=(255, 80, 160)), superpixel_labels)),
                ("lesion_score", heatmap(lesion_score)),
            ],
            columns=2,
        ),
    )

    # Final payload keeps the raw measurements, the friendly summary, and the
    # exported image references in one place for the backend.
    comparison = {
        "baseline_method": baseline["method"],
        "improved_method": improved["method"],
        "baseline_health_status": baseline["health_status"],
        "improved_health_status": improved["health_status"],
        "baseline_infected_area_percentage": baseline["infected_area_percentage"],
        "improved_infected_area_percentage": improved["infected_area_percentage"],
        "infected_area_delta_percentage_points": round(
            improved["infected_area_percentage"] - baseline["infected_area_percentage"], 2
        ),
        "segmentation_agreement": improved["segmentation_agreement"],
        "core_segmentation_methods": list(CORE_SEGMENTATION_METHODS),
        "comparison_only_segmentation_methods": list(COMPARISON_ONLY_SEGMENTATION_METHODS),
        "notes": [
            "Baseline uses single-threshold lesion segmentation.",
            "The deployed improved analysis uses a practical consensus of thresholding, clustering, and graph cut after leaf-first ROI isolation.",
            "Region growing, split-merge, and superpixels are retained as supplementary comparison outputs rather than core decision drivers.",
            "Severity and confidence are derived from consensus lesion coverage, color and texture evidence, and auxiliary frequency cues.",
        ],
    }

    analysis = {
        "pipeline": "plant-vision-cv-v2-leaf-first",
        "input": {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "crop_hint": crop_hint,
        },
        "topics_covered": CORE_TOPICS,
        "supplementary_topics_demonstrated": SUPPLEMENTARY_TOPICS,
        "topic_selection_policy": {
            "core_focus": "Only methods that materially support leaf isolation, lesion measurement, severity estimation, or robust crop-disease classification are treated as core.",
            "supplementary_focus": "Frequency, wavelet, and extra segmentation views are kept as interpretability and comparison aids, not as mandatory textbook checklist items.",
        },
        "preprocessing": {
            "target_resolution": [256, 256],
            "interpolation_methods": ["nearest", "bilinear", "bicubic"],
            "grayscale_mean": round(float(grayscale_array.mean()), 3),
        },
        "roi": {
            "strategy": "leaf_first_localization_and_isolation",
            "leaf_detected": leaf_detected,
            "bbox_xyxy": roi_bbox_xyxy,
            "bbox_normalized_xyxy": roi_bbox_normalized_xyxy,
            "leaf_area_percentage": round(leaf_area_ratio * 100.0, 2),
            "roi_fill_percentage": round(roi_fill_ratio * 100.0, 2),
            "scene_context_relative_path": scene_output["relative_path"],
            "context_crop_relative_path": roi_output["relative_path"],
            "classifier_primary_relative_path": roi_primary_output["relative_path"],
            "notes": [
                "Leaf ROI is localized before disease classification to reduce field-background clutter.",
                "The primary deep-learning input isolates the detected leaf while preserving lesion texture and shape.",
            ],
        },
        "enhancement": {
            "contrast_stretch_applied": True,
            "gamma_correction": 0.9,
            "unsharp_mask_amount": 1.15,
        },
        "restoration": {
            "median_filter_size": 3,
            "wiener_filter_window": 5,
            "reconstruction_note": "Low-pass and high-pass reconstructions are exported in the frequency outputs.",
        },
        "color_analysis": {
            "mean_excess_green": round(float(exg[leaf_mask].mean()) if np.any(leaf_mask) else 0.0, 4),
            "lesion_score_mean": round(float(lesion_score[consensus_mask].mean()) if np.any(consensus_mask) else 0.0, 4),
        },
        "frequency_analysis": {
            "role": "supplementary_interpretability_and_texture_support",
            "high_frequency_ratio": round(float(features["frequency"]["high_to_low_ratio"]), 4),
            "wavelet_energy": {key: round(float(value), 4) for key, value in features["wavelets"].items()},
        },
        "segmentation": {
            "leaf_area_percentage": round(mask_coverage(leaf_mask) * 100.0, 2),
            "deployed_strategy": "leaf_first_consensus_of_thresholding_clustering_and_graph_cut",
            "comparison_only_methods": list(COMPARISON_ONLY_SEGMENTATION_METHODS),
            "methods": {
                name: {
                    "infected_area_percentage": round(mask_coverage(mask, leaf_mask) * 100.0, 2),
                }
                for name, mask in all_method_masks.items()
            },
            "consensus_infected_area_percentage": improved["infected_area_percentage"],
        },
        "features": json_ready(features),
        "postprocessing": {
            "strategy": "connected_component_grouping_on_consensus_lesion_mask",
            "severity_source": "cv_only_consensus_mask_measurement",
            "lesion_region_count": len(lesion_regions),
            "largest_region_percentage_of_leaf": round(float(largest_region_percentage), 2),
            "mean_region_percentage_of_leaf": round(float(mean_region_percentage), 2),
            "regions": lesion_regions,
            "notes": [
                "Marked disease spots are created from connected components on the final consensus lesion mask.",
                "Severity and affected-area values are computed from CV segmentation geometry, not from the deep classifier output.",
            ],
        },
        "classification": {
            "baseline": baseline,
            "improved": improved,
            "comparison": comparison,
        },
        "severity_estimation": {
            "source": "cv_only_consensus_mask_measurement",
            "severity_level": improved["severity_level"],
            "infected_area_percentage": improved["infected_area_percentage"],
            "urgency_level": improved["urgency_level"],
        },
        "outputs": outputs,
        "summary": {
            "predicted_crop": improved["predicted_crop"],
            "health_status": improved["health_status"],
            "predicted_disease": improved["predicted_disease"],
            "confidence_score": improved["confidence_score"],
            "severity_level": improved["severity_level"],
            "urgency_level": improved["urgency_level"],
            "infected_area_percentage": improved["infected_area_percentage"],
        },
    }
    return json_ready(analysis)
