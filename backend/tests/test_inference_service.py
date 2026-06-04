import pytest

from app.services.inference_service import MockInferenceService, fuse_dl_predictions, merge_plant_analysis


@pytest.mark.asyncio
async def test_mock_inference_service_returns_prediction():
    service = MockInferenceService()
    result = await service.predict(b"fake-image-bytes", "leaf.jpg", "image/jpeg")

    assert result.predicted_crop == "tomato"
    assert 0 <= result.confidence_score <= 1
    assert result.raw_prediction_json["pipeline"] == "mock"


def test_merge_plant_analysis_prefers_dl_classifier_for_labels():
    analysis = {
        "pipeline": "plant-vision-cv-v1",
        "summary": {
            "predicted_crop": "tomato",
            "health_status": "diseased",
            "predicted_disease": "leaf_spot_like_pattern",
            "confidence_score": 0.82,
            "severity_level": "high",
            "urgency_level": "urgent",
            "infected_area_percentage": 21.5,
        },
        "classification": {},
    }
    dl_prediction = {
        "predicted_label": "tomato__early_blight",
        "crop": "tomato",
        "disease": "early_blight",
        "is_healthy": False,
        "health_status": "diseased",
        "confidence": 0.93,
        "model_name": "tf_efficientnetv2_s",
    }

    merged, model_version = merge_plant_analysis(
        analysis,
        dl_prediction,
        default_model_version="plant-vision-cv-v1",
    )

    assert merged["summary"]["predicted_disease"] == "early_blight"
    assert merged["summary"]["health_status"] == "diseased"
    assert merged["summary"]["severity_level"] == "high"
    assert merged["summary"]["infected_area_percentage"] == 21.5
    assert merged["model_integration"]["classifier_source"] == "efficientnet_checkpoint"
    assert model_version == "plant-vision-cv-v1+tf_efficientnetv2_s"


def test_merge_plant_analysis_marks_dl_cv_disagreement_as_suspicious():
    analysis = {
        "pipeline": "plant-vision-cv-v1",
        "summary": {
            "predicted_crop": "tomato",
            "health_status": "diseased",
            "predicted_disease": "late_blight_like_pattern",
            "confidence_score": 0.84,
            "severity_level": "medium",
            "urgency_level": "high",
            "infected_area_percentage": 12.0,
        },
        "classification": {},
    }
    dl_prediction = {
        "predicted_label": "tomato__healthy",
        "crop": "tomato",
        "disease": "healthy",
        "is_healthy": True,
        "health_status": "healthy",
        "confidence": 0.98,
        "model_name": "tf_efficientnetv2_s",
    }

    merged, _ = merge_plant_analysis(
        analysis,
        dl_prediction,
        default_model_version="plant-vision-cv-v1",
    )

    assert merged["summary"]["health_status"] == "suspicious"
    assert merged["summary"]["predicted_disease"] is None
    assert merged["model_integration"]["decision_strategy"] == "dl_cv_disagreement_marked_suspicious"


def test_fuse_dl_predictions_prefers_leaf_roi_when_roi_is_strong():
    analysis = {
        "summary": {
            "infected_area_percentage": 11.0,
        },
        "roi": {
            "roi_fill_percentage": 51.0,
        },
    }
    predictions = {
        "full_image": {
            "predicted_label": "tomato__healthy",
            "crop": "tomato",
            "disease": "healthy",
            "is_healthy": True,
            "health_status": "healthy",
            "confidence": 0.84,
            "model_name": "tf_efficientnetv2_s",
        },
        "leaf_roi": {
            "predicted_label": "tomato__early_blight",
            "crop": "tomato",
            "disease": "early_blight",
            "is_healthy": False,
            "health_status": "diseased",
            "confidence": 0.82,
            "model_name": "tf_efficientnetv2_s",
        },
    }

    fused = fuse_dl_predictions(predictions, analysis)

    assert fused is not None
    assert fused["predicted_label"] == "tomato__early_blight"
    assert fused["branch_used"] == "leaf_roi"
    assert fused["fusion_strategy"] in {
        "prefer_leaf_roi_on_disagreement",
        "cv_override_prefers_diseased_branch",
        "default_leaf_roi_priority",
    }


def test_fuse_dl_predictions_can_keep_full_image_when_margin_is_large():
    analysis = {
        "summary": {
            "infected_area_percentage": 1.5,
        },
        "roi": {
            "roi_fill_percentage": 18.0,
        },
    }
    predictions = {
        "full_image": {
            "predicted_label": "corn__common_rust",
            "crop": "corn",
            "disease": "common_rust",
            "is_healthy": False,
            "health_status": "diseased",
            "confidence": 0.93,
            "model_name": "tf_efficientnetv2_s",
        },
        "leaf_roi": {
            "predicted_label": "corn__healthy",
            "crop": "corn",
            "disease": "healthy",
            "is_healthy": True,
            "health_status": "healthy",
            "confidence": 0.72,
            "model_name": "tf_efficientnetv2_s",
        },
    }

    fused = fuse_dl_predictions(predictions, analysis)

    assert fused is not None
    assert fused["predicted_label"] == "corn__common_rust"
    assert fused["branch_used"] == "full_image"
    assert fused["fusion_strategy"] == "prefer_full_image_high_margin"


def test_fuse_dl_predictions_uses_weighted_probability_fusion_when_available():
    analysis = {
        "summary": {
            "infected_area_percentage": 8.0,
        },
        "roi": {
            "roi_fill_percentage": 42.0,
        },
    }
    predictions = {
        "full_image": {
            "predicted_label": "tomato__healthy",
            "crop": "tomato",
            "disease": "healthy",
            "is_healthy": True,
            "health_status": "healthy",
            "confidence": 0.61,
            "model_name": "tf_efficientnetv2_s",
            "class_probabilities": {
                "tomato__healthy": 0.61,
                "tomato__early_blight": 0.39,
            },
        },
        "leaf_roi": {
            "predicted_label": "tomato__early_blight",
            "crop": "tomato",
            "disease": "early_blight",
            "is_healthy": False,
            "health_status": "diseased",
            "confidence": 0.64,
            "model_name": "tf_efficientnetv2_s",
            "class_probabilities": {
                "tomato__healthy": 0.20,
                "tomato__early_blight": 0.80,
            },
        },
    }

    fused = fuse_dl_predictions(predictions, analysis)

    assert fused is not None
    assert fused["predicted_label"] == "tomato__early_blight"
    assert fused["branch_used"] == "probability_fusion"
    assert fused["fusion_strategy"] == "weighted_probability_fusion_leaf_roi"
