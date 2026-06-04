"""Plant-image segmentation helpers covering multiple DIP methods."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse.csgraph import maximum_flow
from sklearn.cluster import KMeans

from ml.common.preprocessing import normalize_rgb, rgb_to_hsv_array
from ml.common.transforms import compute_brownness_index, compute_excess_green
from ml.common.utils import min_max_normalize


def otsu_threshold_mask(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    cumulative_weight = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * np.arange(256))
    global_mean = cumulative_mean[-1]

    denominator = cumulative_weight * (total - cumulative_weight)
    denominator[denominator == 0] = 1
    between_class = ((global_mean * cumulative_weight - cumulative_mean) ** 2) / denominator
    threshold = int(np.argmax(between_class))
    return (gray >= threshold).astype(np.uint8) * 255


def _otsu_threshold_from_values(values: np.ndarray) -> float:
    scaled = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    histogram = np.bincount(scaled, minlength=256).astype(np.float64)
    total = scaled.size
    cumulative_weight = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * np.arange(256))
    global_mean = cumulative_mean[-1]

    denominator = cumulative_weight * (total - cumulative_weight)
    denominator[denominator == 0] = 1
    between_class = ((global_mean * cumulative_weight - cumulative_mean) ** 2) / denominator
    threshold = int(np.argmax(between_class))
    return threshold / 255.0


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    largest_label = int(np.argmax(sizes)) + 1
    return labels == largest_label


def _mask_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        return None
    top = int(rows.min())
    bottom = int(rows.max()) + 1
    left = int(cols.min())
    right = int(cols.max()) + 1
    return top, left, bottom, right


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    pil_mask = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = pil_mask.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def _component_score(
    component_mask: np.ndarray,
    exg_map: np.ndarray,
    hsv_array: np.ndarray,
    brownness_map: np.ndarray,
) -> float:
    area_ratio = float(component_mask.mean())
    if area_ratio <= 0:
        return 0.0

    rows, cols = np.where(component_mask)
    row_center = float(rows.mean())
    col_center = float(cols.mean())
    center_row = (component_mask.shape[0] - 1) / 2.0
    center_col = (component_mask.shape[1] - 1) / 2.0
    norm_distance = np.sqrt(
        ((row_center - center_row) / max(center_row, 1.0)) ** 2
        + ((col_center - center_col) / max(center_col, 1.0)) ** 2
    )
    center_score = max(0.0, 1.0 - min(norm_distance, 1.5) / 1.5)

    exg_score = float(exg_map[component_mask].mean()) if np.any(component_mask) else 0.0
    saturation_score = float(hsv_array[..., 1][component_mask].mean()) if np.any(component_mask) else 0.0
    brownness_score = float(brownness_map[component_mask].mean()) if np.any(component_mask) else 0.0
    bbox = _mask_bounding_box(component_mask)
    if bbox is None:
        return 0.0
    top, left, bottom, right = bbox
    bbox_area = max((bottom - top) * (right - left), 1)
    fill_ratio = float(component_mask[top:bottom, left:right].sum()) / float(bbox_area)

    return (
        0.32 * min(1.0, exg_score)
        + 0.18 * min(1.0, saturation_score * 1.6)
        + 0.12 * min(1.0, brownness_score * 1.3)
        + 0.20 * min(1.0, np.sqrt(area_ratio) * 1.8)
        + 0.10 * center_score
        + 0.08 * fill_ratio
    )


def _cluster_leaf_candidate(image_array: np.ndarray) -> np.ndarray:
    rows, cols = image_array.shape[:2]
    target_side = 176
    if max(rows, cols) > target_side:
        resized_image = np.asarray(
            Image.fromarray(image_array.astype(np.uint8)).resize(
                (target_side, target_side), Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )
    else:
        resized_image = image_array.astype(np.uint8)

    rgb = normalize_rgb(resized_image)
    hsv = rgb_to_hsv_array(rgb)
    exg = min_max_normalize(compute_excess_green(rgb))
    brownness = min_max_normalize(compute_brownness_index(rgb))
    yy, xx = np.mgrid[0 : resized_image.shape[0], 0 : resized_image.shape[1]]
    yy = yy.astype(np.float32) / max(resized_image.shape[0] - 1, 1)
    xx = xx.astype(np.float32) / max(resized_image.shape[1] - 1, 1)

    features = np.column_stack(
        [
            rgb.reshape(-1, 3),
            exg.reshape(-1, 1),
            hsv[..., 1].reshape(-1, 1),
            brownness.reshape(-1, 1),
            yy.reshape(-1, 1),
            xx.reshape(-1, 1),
        ]
    )

    cluster_count = 4 if features.shape[0] >= 1024 else 3
    clustering = KMeans(n_clusters=cluster_count, n_init=5, random_state=11)
    labels = clustering.fit_predict(features)

    candidate_mask_small = np.zeros(resized_image.shape[:2], dtype=bool)
    for cluster_id in range(cluster_count):
        cluster_pixels = labels == cluster_id
        if not np.any(cluster_pixels):
            continue
        cluster_area = float(cluster_pixels.mean())
        cluster_exg = float(exg.reshape(-1)[cluster_pixels].mean())
        cluster_sat = float(hsv[..., 1].reshape(-1)[cluster_pixels].mean())
        cluster_brown = float(brownness.reshape(-1)[cluster_pixels].mean())
        cluster_y = yy.reshape(-1)[cluster_pixels].mean()
        cluster_x = xx.reshape(-1)[cluster_pixels].mean()
        center_distance = np.sqrt((cluster_y - 0.5) ** 2 + (cluster_x - 0.5) ** 2)
        center_score = max(0.0, 1.0 - min(center_distance / 0.75, 1.0))
        cluster_score = (
            0.34 * cluster_exg
            + 0.18 * cluster_sat
            + 0.16 * cluster_brown
            + 0.16 * min(1.0, np.sqrt(cluster_area) * 2.0)
            + 0.16 * center_score
        )
        if cluster_score >= 0.26:
            candidate_mask_small |= cluster_pixels.reshape(resized_image.shape[:2])

    candidate_mask_small = ndimage.binary_closing(candidate_mask_small, structure=np.ones((5, 5), dtype=bool))
    candidate_mask_small = ndimage.binary_fill_holes(candidate_mask_small)
    if candidate_mask_small.mean() <= 0:
        return np.zeros((rows, cols), dtype=bool)
    return _resize_mask(candidate_mask_small, (rows, cols))


def build_leaf_mask(image_array: np.ndarray) -> np.ndarray:
    rgb = normalize_rgb(image_array)
    hsv = rgb_to_hsv_array(rgb)
    exg = compute_excess_green(rgb)
    exg_norm = min_max_normalize(exg)
    brownness = min_max_normalize(compute_brownness_index(rgb))
    green_dominance = min_max_normalize(rgb[..., 1] - 0.5 * (rgb[..., 0] + rgb[..., 2]))

    base_mask = (
        ((exg_norm > max(float(np.percentile(exg_norm, 58)), 0.30)) | (green_dominance > 0.45))
        & (hsv[..., 1] > 0.14)
        & (hsv[..., 2] > 0.10)
    )
    stressed_leaf_mask = (
        (brownness > max(float(np.percentile(brownness, 68)), 0.34))
        & (hsv[..., 1] > 0.15)
        & (hsv[..., 2] > 0.10)
    )
    cluster_mask = _cluster_leaf_candidate(image_array)

    candidate = base_mask | stressed_leaf_mask | cluster_mask
    if candidate.mean() < 0.05:
        fallback_threshold = max(float(np.percentile(exg_norm, 48)), 0.22)
        candidate = ((exg_norm > fallback_threshold) | cluster_mask) & (hsv[..., 2] > 0.08)

    # Reject clearly non-plant colours so coloured backdrops (purple / blue /
    # magenta studio paper, etc.) cannot leak into the mask. Plant tissue is never
    # a saturated blue/purple/magenta. Guarded so it can't wipe a blue-cast image.
    non_plant_color = (hsv[..., 0] > 0.5) & (hsv[..., 0] < 0.93) & (hsv[..., 1] > 0.12)
    if (candidate & ~non_plant_color).mean() > 0.01:
        candidate = candidate & ~non_plant_color

    # Anchor on the green leaf body. The brown/stressed and cluster cues above also
    # fire on reddish/brown *backgrounds* — a brown background pixel is the same
    # colour as a brown lesion, so colour alone cannot separate them. Instead of
    # trusting the raw union (which let backgrounds leak in and then get flagged as
    # lesions downstream, since every lesion mask is intersected with this one), we
    # build the leaf silhouette from the connected green body and fill its interior:
    # interior lesions become filled holes and are kept, while background regions
    # outside the leaf body are dropped even when brown/red. Falls back to the raw
    # union only when there is essentially no green to anchor on (fully necrotic leaf).
    green_anchor = ndimage.binary_opening(base_mask, structure=np.ones((3, 3), dtype=bool))
    green_anchor = ndimage.binary_closing(green_anchor, structure=np.ones((7, 7), dtype=bool))
    if green_anchor.mean() >= 0.02:
        green_anchor = _largest_component(green_anchor)
        silhouette = ndimage.binary_closing(green_anchor, structure=np.ones((9, 9), dtype=bool))
        silhouette = ndimage.binary_fill_holes(silhouette)
        mask = silhouette
    else:
        mask = candidate

    mask = ndimage.binary_closing(mask, structure=np.ones((7, 7), dtype=bool))
    mask = ndimage.binary_opening(mask, structure=np.ones((5, 5), dtype=bool))
    mask = ndimage.binary_fill_holes(mask)

    labels, count = ndimage.label(mask)
    if count == 0:
        return mask

    best_score = -1.0
    best_mask = None
    for label_id in range(1, count + 1):
        component_mask = labels == label_id
        score = _component_score(component_mask, exg_norm, hsv, brownness)
        if score > best_score:
            best_score = score
            best_mask = component_mask

    if best_mask is None:
        return _largest_component(mask)
    return best_mask


def extract_leaf_roi(
    image_array: np.ndarray,
    leaf_mask: np.ndarray,
    *,
    padding_ratio: float = 0.12,
    isolated_background: int = 255,
) -> dict[str, Any]:
    bbox = _mask_bounding_box(leaf_mask)
    image_height, image_width = image_array.shape[:2]
    if bbox is None:
        isolated = np.full_like(image_array, isolated_background, dtype=np.uint8)
        return {
            "leaf_detected": False,
            "bbox_xyxy": [0, 0, image_width, image_height],
            "bbox_normalized_xyxy": [0.0, 0.0, 1.0, 1.0],
            "context_crop": image_array.copy(),
            "isolated_crop": isolated,
            "crop_mask": np.zeros((image_height, image_width), dtype=bool),
            "leaf_area_ratio": 0.0,
            "roi_fill_ratio": 0.0,
        }

    top, left, bottom, right = bbox
    height = max(bottom - top, 1)
    width = max(right - left, 1)
    pad_y = max(4, int(round(height * padding_ratio)))
    pad_x = max(4, int(round(width * padding_ratio)))

    top = max(0, top - pad_y)
    left = max(0, left - pad_x)
    bottom = min(image_height, bottom + pad_y)
    right = min(image_width, right + pad_x)

    context_crop = image_array[top:bottom, left:right].copy()
    crop_mask = leaf_mask[top:bottom, left:right]
    isolated_crop = np.full_like(context_crop, isolated_background, dtype=np.uint8)
    isolated_crop[crop_mask] = context_crop[crop_mask]

    bbox_area = max((bottom - top) * (right - left), 1)
    leaf_pixels = int(crop_mask.sum())
    return {
        "leaf_detected": leaf_pixels > 0,
        "bbox_xyxy": [int(left), int(top), int(right), int(bottom)],
        "bbox_normalized_xyxy": [
            round(float(left) / max(image_width, 1), 4),
            round(float(top) / max(image_height, 1), 4),
            round(float(right) / max(image_width, 1), 4),
            round(float(bottom) / max(image_height, 1), 4),
        ],
        "context_crop": context_crop,
        "isolated_crop": isolated_crop,
        "crop_mask": crop_mask,
        "leaf_area_ratio": round(float(leaf_mask.mean()), 4),
        "roi_fill_ratio": round(float(leaf_pixels) / float(bbox_area), 4),
    }


def lesion_score_map(image_array: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    rgb = normalize_rgb(image_array)
    hsv = rgb_to_hsv_array(rgb)
    exg = compute_excess_green(rgb)
    brownness = compute_brownness_index(rgb)
    red_green_gap = rgb[..., 0] - rgb[..., 1]
    local_texture = ndimage.gaussian_laplace(rgb.mean(axis=-1), sigma=1.1)

    score = (
        0.30 * min_max_normalize(red_green_gap)
        + 0.25 * min_max_normalize(-exg)
        + 0.20 * min_max_normalize(brownness)
        + 0.15 * min_max_normalize(hsv[..., 1])
        + 0.10 * min_max_normalize(np.abs(local_texture))
    )
    score = min_max_normalize(score)
    return score * leaf_mask


def threshold_segmentation(image_array: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    lesion_score = lesion_score_map(image_array, leaf_mask)
    pixels = lesion_score[leaf_mask]
    if pixels.size == 0:
        return np.zeros_like(leaf_mask, dtype=bool)
    threshold = _otsu_threshold_from_values(pixels)
    mask = (lesion_score >= threshold) & leaf_mask
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
    return mask


def region_growing_segmentation(image_array: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    lesion_score = lesion_score_map(image_array, leaf_mask)
    if not np.any(leaf_mask):
        return np.zeros_like(leaf_mask, dtype=bool)

    seed_index = np.argmax(lesion_score * leaf_mask)
    rows, cols = leaf_mask.shape
    seed_row, seed_col = divmod(int(seed_index), cols)
    seed_value = float(lesion_score[seed_row, seed_col])
    if seed_value <= 0:
        return np.zeros_like(leaf_mask, dtype=bool)

    rgb = normalize_rgb(image_array)
    seed_color = rgb[seed_row, seed_col]
    value_threshold = max(0.32, seed_value * 0.68)
    queue: deque[tuple[int, int]] = deque([(seed_row, seed_col)])
    visited = np.zeros_like(leaf_mask, dtype=bool)
    region = np.zeros_like(leaf_mask, dtype=bool)

    while queue:
        row, col = queue.popleft()
        if visited[row, col]:
            continue
        visited[row, col] = True
        if not leaf_mask[row, col]:
            continue

        color_distance = np.linalg.norm(rgb[row, col] - seed_color)
        if lesion_score[row, col] >= value_threshold and color_distance <= 0.42:
            region[row, col] = True
            for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= next_row < rows and 0 <= next_col < cols and not visited[next_row, next_col]:
                    queue.append((next_row, next_col))

    region = ndimage.binary_closing(region, structure=np.ones((5, 5), dtype=bool))
    return region & leaf_mask


def split_and_merge_segmentation(image_array: np.ndarray, leaf_mask: np.ndarray, min_block_size: int = 16) -> np.ndarray:
    lesion_score = lesion_score_map(image_array, leaf_mask)
    result = np.zeros_like(leaf_mask, dtype=bool)

    def recurse(row_start: int, row_end: int, col_start: int, col_end: int) -> None:
        block_mask = leaf_mask[row_start:row_end, col_start:col_end]
        if block_mask.sum() == 0:
            return
        block_scores = lesion_score[row_start:row_end, col_start:col_end][block_mask]
        if block_scores.size == 0:
            return

        block_height = row_end - row_start
        block_width = col_end - col_start
        mean_score = float(block_scores.mean())
        variance = float(block_scores.var())

        if min(block_height, block_width) <= min_block_size or variance < 0.006:
            if mean_score >= 0.54:
                result[row_start:row_end, col_start:col_end] |= block_mask
            return

        row_mid = row_start + block_height // 2
        col_mid = col_start + block_width // 2
        recurse(row_start, row_mid, col_start, col_mid)
        recurse(row_start, row_mid, col_mid, col_end)
        recurse(row_mid, row_end, col_start, col_mid)
        recurse(row_mid, row_end, col_mid, col_end)

    recurse(0, leaf_mask.shape[0], 0, leaf_mask.shape[1])
    result = ndimage.binary_opening(result, structure=np.ones((3, 3), dtype=bool))
    result = ndimage.binary_closing(result, structure=np.ones((5, 5), dtype=bool))
    return result & leaf_mask


def clustering_segmentation(image_array: np.ndarray, leaf_mask: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    rgb = normalize_rgb(image_array)
    lesion_score = lesion_score_map(image_array, leaf_mask)
    rows, cols = leaf_mask.shape
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]

    pixels = leaf_mask.reshape(-1)
    if pixels.sum() == 0:
        return np.zeros_like(leaf_mask, dtype=bool)

    features = np.column_stack(
        [
            rgb.reshape(-1, 3)[pixels],
            lesion_score.reshape(-1, 1)[pixels],
            (y_coords.reshape(-1, 1)[pixels] / max(rows - 1, 1)),
            (x_coords.reshape(-1, 1)[pixels] / max(cols - 1, 1)),
        ]
    )
    clustering = KMeans(n_clusters=n_clusters, n_init=10, random_state=7)
    labels = clustering.fit_predict(features)

    cluster_scores = []
    for cluster_id in range(n_clusters):
        cluster_scores.append(float(features[labels == cluster_id, 3].mean()))
    lesion_cluster = int(np.argmax(cluster_scores))

    mask = np.zeros(rows * cols, dtype=bool)
    mask[np.flatnonzero(pixels)[labels == lesion_cluster]] = True
    mask = mask.reshape(rows, cols)
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    return mask & leaf_mask


def simple_superpixels(
    image_array: np.ndarray,
    leaf_mask: np.ndarray,
    n_segments: int = 64,
    compactness: float = 0.16,
    iterations: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = normalize_rgb(image_array)
    rows, cols, _ = rgb.shape
    step = max(6, int(np.sqrt((rows * cols) / max(n_segments, 1))))
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]

    centers: list[np.ndarray] = []
    for row in range(step // 2, rows, step):
        for col in range(step // 2, cols, step):
            if leaf_mask[row, col]:
                centers.append(np.array([row, col, *rgb[row, col]], dtype=np.float32))

    if not centers:
        labels = np.zeros((rows, cols), dtype=np.int32)
        return labels, np.zeros_like(leaf_mask, dtype=bool)

    centers_array = np.vstack(centers)
    distances = np.full((rows, cols), np.inf, dtype=np.float32)
    labels = -np.ones((rows, cols), dtype=np.int32)

    for _ in range(iterations):
        for center_id, center in enumerate(centers_array):
            row, col, red, green, blue = center
            row_start = max(int(row - step), 0)
            row_end = min(int(row + step), rows)
            col_start = max(int(col - step), 0)
            col_end = min(int(col + step), cols)

            region_rgb = rgb[row_start:row_end, col_start:col_end]
            region_y = y_coords[row_start:row_end, col_start:col_end]
            region_x = x_coords[row_start:row_end, col_start:col_end]
            color_distance = np.sqrt(
                (region_rgb[..., 0] - red) ** 2
                + (region_rgb[..., 1] - green) ** 2
                + (region_rgb[..., 2] - blue) ** 2
            )
            space_distance = np.sqrt((region_y - row) ** 2 + (region_x - col) ** 2)
            combined = color_distance + compactness * space_distance / max(step, 1)

            window = distances[row_start:row_end, col_start:col_end]
            replace = combined < window
            window[replace] = combined[replace]
            labels[row_start:row_end, col_start:col_end][replace] = center_id

        for center_id in range(len(centers_array)):
            region = labels == center_id
            if not np.any(region):
                continue
            centers_array[center_id, 0] = y_coords[region].mean()
            centers_array[center_id, 1] = x_coords[region].mean()
            centers_array[center_id, 2:] = rgb[region].mean(axis=0)

    lesion_score = lesion_score_map(image_array, leaf_mask)
    lesion_mask = np.zeros_like(leaf_mask, dtype=bool)
    for center_id in range(len(centers_array)):
        region = labels == center_id
        region_leaf = region & leaf_mask
        if region_leaf.sum() == 0:
            continue
        if float(lesion_score[region_leaf].mean()) >= 0.53:
            lesion_mask |= region_leaf

    return labels, lesion_mask


def graph_cut_segmentation(image_array: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    rgb = normalize_rgb(image_array)
    lesion_score = lesion_score_map(image_array, leaf_mask)
    original_rows, original_cols = leaf_mask.shape
    target_size = 72

    resized_rgb = np.asarray(
        Image.fromarray((rgb * 255).astype(np.uint8)).resize((target_size, target_size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    resized_mask = (
        np.asarray(
            Image.fromarray((leaf_mask.astype(np.uint8) * 255)).resize((target_size, target_size), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        > 0
    )
    resized_score = np.asarray(
        Image.fromarray((lesion_score * 255).astype(np.uint8)).resize((target_size, target_size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    exg = min_max_normalize(compute_excess_green(resized_rgb))
    lesion_seed = resized_mask & (resized_score >= np.percentile(resized_score[resized_mask], 88) if np.any(resized_mask) else False)
    healthy_seed = resized_mask & (exg >= np.percentile(exg[resized_mask], 75) if np.any(resized_mask) else False) & (resized_score <= 0.35)

    rows, cols = resized_mask.shape
    source = rows * cols
    sink = source + 1
    graph = sparse.lil_matrix((rows * cols + 2, rows * cols + 2), dtype=np.int32)

    def node_index(row: int, col: int) -> int:
        return row * cols + col

    for row in range(rows):
        for col in range(cols):
            node = node_index(row, col)
            score = float(resized_score[row, col])
            healthy_evidence = int(10 + 90 * float(exg[row, col]))
            lesion_evidence = int(10 + 90 * score)

            if not resized_mask[row, col]:
                healthy_evidence += 220
                lesion_evidence = 0
            if healthy_seed[row, col]:
                healthy_evidence += 260
            if lesion_seed[row, col]:
                lesion_evidence += 260

            graph[source, node] = healthy_evidence
            graph[node, sink] = lesion_evidence

            for next_row, next_col in ((row + 1, col), (row, col + 1)):
                if next_row >= rows or next_col >= cols:
                    continue
                next_node = node_index(next_row, next_col)
                color_difference = np.linalg.norm(resized_rgb[row, col] - resized_rgb[next_row, next_col])
                smoothness = int(max(4.0, 36.0 * np.exp(-4.0 * color_difference)))
                graph[node, next_node] = smoothness
                graph[next_node, node] = smoothness

    flow = maximum_flow(graph.tocsr(), source, sink)
    residual = (graph.tocsr() - flow.flow).tolil()

    reachable = np.zeros(rows * cols + 2, dtype=bool)
    queue: deque[int] = deque([source])
    reachable[source] = True
    while queue:
        node = queue.popleft()
        neighbors = residual.rows[node]
        capacities = residual.data[node]
        for neighbor, capacity in zip(neighbors, capacities, strict=False):
            if capacity > 0 and not reachable[neighbor]:
                reachable[neighbor] = True
                queue.append(neighbor)

    cut_mask = np.zeros((rows, cols), dtype=bool)
    for row in range(rows):
        for col in range(cols):
            node = node_index(row, col)
            cut_mask[row, col] = resized_mask[row, col] and not reachable[node]

    upsampled = (
        np.asarray(
            Image.fromarray((cut_mask.astype(np.uint8) * 255)).resize((original_cols, original_rows), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        > 0
    )
    upsampled = ndimage.binary_opening(upsampled, structure=np.ones((3, 3), dtype=bool))
    upsampled = ndimage.binary_closing(upsampled, structure=np.ones((5, 5), dtype=bool))
    return upsampled & leaf_mask
