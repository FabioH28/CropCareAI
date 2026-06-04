"""Canonical label schema and dataset-specific label normalization."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LabelInfo:
    canonical_label: str
    crop: str
    disease: str
    is_healthy: bool


CANONICAL_LABELS: dict[str, LabelInfo] = {
    "apple__apple_scab": LabelInfo("apple__apple_scab", "apple", "apple_scab", False),
    "apple__black_rot": LabelInfo("apple__black_rot", "apple", "black_rot", False),
    "apple__cedar_apple_rust": LabelInfo("apple__cedar_apple_rust", "apple", "cedar_apple_rust", False),
    "apple__healthy": LabelInfo("apple__healthy", "apple", "healthy", True),
    "corn__common_rust": LabelInfo("corn__common_rust", "corn", "common_rust", False),
    "corn__gray_leaf_spot": LabelInfo("corn__gray_leaf_spot", "corn", "gray_leaf_spot", False),
    "corn__healthy": LabelInfo("corn__healthy", "corn", "healthy", True),
    "corn__northern_leaf_blight": LabelInfo("corn__northern_leaf_blight", "corn", "northern_leaf_blight", False),
    "grape__black_rot": LabelInfo("grape__black_rot", "grape", "black_rot", False),
    "grape__esca_black_measles": LabelInfo("grape__esca_black_measles", "grape", "esca_black_measles", False),
    "grape__healthy": LabelInfo("grape__healthy", "grape", "healthy", True),
    "grape__leaf_blight": LabelInfo("grape__leaf_blight", "grape", "leaf_blight", False),
    "pepper__bacterial_spot": LabelInfo("pepper__bacterial_spot", "pepper", "bacterial_spot", False),
    "pepper__healthy": LabelInfo("pepper__healthy", "pepper", "healthy", True),
    "potato__early_blight": LabelInfo("potato__early_blight", "potato", "early_blight", False),
    "potato__healthy": LabelInfo("potato__healthy", "potato", "healthy", True),
    "potato__late_blight": LabelInfo("potato__late_blight", "potato", "late_blight", False),
    "tomato__bacterial_spot": LabelInfo("tomato__bacterial_spot", "tomato", "bacterial_spot", False),
    "tomato__early_blight": LabelInfo("tomato__early_blight", "tomato", "early_blight", False),
    "tomato__healthy": LabelInfo("tomato__healthy", "tomato", "healthy", True),
    "tomato__late_blight": LabelInfo("tomato__late_blight", "tomato", "late_blight", False),
    "tomato__leaf_mold": LabelInfo("tomato__leaf_mold", "tomato", "leaf_mold", False),
    "tomato__septoria_leaf_spot": LabelInfo("tomato__septoria_leaf_spot", "tomato", "septoria_leaf_spot", False),
    "tomato__spider_mites": LabelInfo("tomato__spider_mites", "tomato", "spider_mites", False),
    "tomato__target_spot": LabelInfo("tomato__target_spot", "tomato", "target_spot", False),
    "tomato__mosaic_virus": LabelInfo("tomato__mosaic_virus", "tomato", "mosaic_virus", False),
    "tomato__yellow_leaf_curl_virus": LabelInfo("tomato__yellow_leaf_curl_virus", "tomato", "yellow_leaf_curl_virus", False),
}

PLANTVILLAGE_ALIASES = {
    "Apple___Apple_scab": "apple__apple_scab",
    "Apple___Black_rot": "apple__black_rot",
    "Apple___Cedar_apple_rust": "apple__cedar_apple_rust",
    "Apple___healthy": "apple__healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "corn__gray_leaf_spot",
    "Corn___Common_rust": "corn__common_rust",
    "Corn___healthy": "corn__healthy",
    "Corn___Northern_Leaf_Blight": "corn__northern_leaf_blight",
    "Grape___Black_rot": "grape__black_rot",
    "Grape___Esca_(Black_Measles)": "grape__esca_black_measles",
    "Grape___healthy": "grape__healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape__leaf_blight",
    "Pepper,_bell___Bacterial_spot": "pepper__bacterial_spot",
    "Pepper,_bell___healthy": "pepper__healthy",
    "Potato___Early_blight": "potato__early_blight",
    "Potato___healthy": "potato__healthy",
    "Potato___Late_blight": "potato__late_blight",
    "Tomato___Bacterial_spot": "tomato__bacterial_spot",
    "Tomato___Early_blight": "tomato__early_blight",
    "Tomato___healthy": "tomato__healthy",
    "Tomato___Late_blight": "tomato__late_blight",
    "Tomato___Leaf_Mold": "tomato__leaf_mold",
    "Tomato___Septoria_leaf_spot": "tomato__septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato__spider_mites",
    "Tomato___Target_Spot": "tomato__target_spot",
    "Tomato___Tomato_mosaic_virus": "tomato__mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato__yellow_leaf_curl_virus",
}

PLANTDOC_ALIASES = {
    "Apple leaf": "apple__healthy",
    "Apple rust leaf": "apple__cedar_apple_rust",
    "Apple Scab Leaf": "apple__apple_scab",
    "Bell_pepper leaf": "pepper__healthy",
    "Bell_pepper leaf spot": "pepper__bacterial_spot",
    "Corn Gray leaf spot": "corn__gray_leaf_spot",
    "Corn leaf blight": "corn__northern_leaf_blight",
    "Corn rust leaf": "corn__common_rust",
    "grape leaf": "grape__healthy",
    "grape leaf black rot": "grape__black_rot",
    "Potato leaf early blight": "potato__early_blight",
    "Potato leaf late blight": "potato__late_blight",
    "Tomato Early blight leaf": "tomato__early_blight",
    "Tomato leaf": "tomato__healthy",
    "Tomato leaf bacterial spot": "tomato__bacterial_spot",
    "Tomato leaf late blight": "tomato__late_blight",
    "Tomato leaf mosaic virus": "tomato__mosaic_virus",
    "Tomato leaf yellow virus": "tomato__yellow_leaf_curl_virus",
    "Tomato mold leaf": "tomato__leaf_mold",
    "Tomato Septoria leaf spot": "tomato__septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "tomato__spider_mites",
}

MENDELEY_TOMATO_ALIASES = {
    "Tomato___Bacterial_spot": "tomato__bacterial_spot",
    "Tomato___Early_blight": "tomato__early_blight",
    "Tomato___healthy": "tomato__healthy",
    "Tomato___Late_blight": "tomato__late_blight",
    "Tomato___Leaf_Mold": "tomato__leaf_mold",
    "Tomato___Septoria_leaf_spot": "tomato__septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato__spider_mites",
    "Tomato___Target_Spot": "tomato__target_spot",
    "Tomato___Tomato_mosaic_virus": "tomato__mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato__yellow_leaf_curl_virus",
    "Tomato Early blight": "tomato__early_blight",
    "Tomato healthy": "tomato__healthy",
    "Tomato late blight": "tomato__late_blight",
    "Tomato leaf mold": "tomato__leaf_mold",
    "Tomato septoria leaf spot": "tomato__septoria_leaf_spot",
    "Tomato spider mites": "tomato__spider_mites",
    "Tomato target spot": "tomato__target_spot",
    "Tomato mosaic virus": "tomato__mosaic_virus",
    "Tomato yellow leaf curl virus": "tomato__yellow_leaf_curl_virus",
    "Tomato bacterial spot": "tomato__bacterial_spot",
}

PLANT_PATHOLOGY_2021_ALIASES = {
    # Plant Pathology 2021 ships multi-label rows (space-separated). We only
    # accept rows whose label is exactly one of the three apple classes already
    # in our canonical schema. Multi-label rows ("scab frog_eye_leaf_spot",
    # "complex", etc.) are skipped by the manifest builder.
    "healthy": "apple__healthy",
    "scab": "apple__apple_scab",
    "rust": "apple__cedar_apple_rust",
}

PLANT_PATHOLOGY_2020_ALIASES = {
    # Plant Pathology 2020 (FGVC7) uses one-hot columns: healthy,
    # multiple_diseases, scab, rust. The manifest builder converts each row to
    # a single label and keeps only the three classes already in our schema.
    # Rows where multiple_diseases=1 (or any multi-class row) are skipped.
    "healthy": "apple__healthy",
    "scab": "apple__apple_scab",
    "rust": "apple__cedar_apple_rust",
}

DATASET_ALIASES = {
    "plantvillage": PLANTVILLAGE_ALIASES,
    "plantdoc": PLANTDOC_ALIASES,
    "mendeley_tomato": MENDELEY_TOMATO_ALIASES,
    "plant_pathology_2021": PLANT_PATHOLOGY_2021_ALIASES,
    "plant_pathology_2020": PLANT_PATHOLOGY_2020_ALIASES,
}


def normalize_label(dataset_name: str, raw_label: str) -> LabelInfo | None:
    alias_map = DATASET_ALIASES.get(dataset_name, {})
    canonical = alias_map.get(raw_label)
    if canonical is None:
        return None
    return CANONICAL_LABELS[canonical]
