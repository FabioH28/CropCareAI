# CropCare AI Plant Module

This folder is now split so you can tell the role of each part immediately.

## Structure

- `cv/`: computer vision and digital image processing
- `dl/`: deep-learning model training and inference
- `llm/`: future language-model explanation/reporting layer
- `artifacts/`: manifests, checkpoints, reports, and analysis outputs

## What Each Layer Does

- `cv/`: preprocessing, enhancement, restoration, color analysis, leaf-first ROI isolation, core lesion segmentation, handcrafted features, infected-area estimation, severity estimation, and comparison visuals
- `dl/`: dataset preparation, EfficientNet-based disease classifier training, checkpointed inference, and evaluation
- `llm/`: placeholder helpers for turning analysis output into farmer-friendly explanations

## What It Uses Right Now

- `datasets/plant/plantvillage/Plant_leave_diseases_dataset_without_augmentation`
- `datasets/plant/plantdoc`

## What It Can Use Later

- `datasets/plant/tomato-leaves/Dataset of Tomato Leaves`

That tomato dataset is currently still archived as `.7z`, so the pipeline is written to skip it cleanly until you extract it.

## Training Strategy

This is designed for higher real-world accuracy on taken photos, not just clean benchmark accuracy.

### Stage 1

Train on controlled images from PlantVillage.

### Stage 2

Fine-tune on PlantDoc field images plus a small replay slice of PlantVillage data to preserve class stability.

### Final Evaluation

- controlled test set from PlantVillage
- real-world field test set from PlantDoc

## Supported Focus Crops

- apple
- corn
- grape
- pepper
- potato
- tomato

## Install

```powershell
cd "c:\Users\Admin\Desktop\School Projects\Farm Plant Disease Detection and Treatment Advisor\ml\plant"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Build the Dataset Manifest

```powershell
python dl/prepare_dataset.py
```

## Train the Model

```powershell
python dl/train.py --model-name tf_efficientnetv2_s --image-size 300 --batch-size 16 --epochs-stage1 8 --epochs-stage2 5 --output-name tf_efficientnetv2_s_main --promote-to-production
```

## Run a Backbone Sweep and Promote the Winner

```powershell
python ..\experiments\plant\run_model_sweep.py --config "..\experiments\plant\configs\rtx3050ti_high_accuracy.json"
```

This writes one checkpoint per backbone, ranks them with field performance first, and copies the winning model into the backend-facing production folder.

For a stronger cloud run:

```powershell
python ..\experiments\plant\run_model_sweep.py --config "..\experiments\plant\configs\cloud_high_accuracy.json"
```

## Run Inference

```powershell
python dl/infer.py "C:\path\to\leaf.jpg" --checkpoint "artifacts\production\best_model.pt" --tta-passes 2
```

## Check If Your Environment Is GPU-Ready

```powershell
python ..\experiments\plant\check_training_env.py
```

## Run The Full Plant CV Analysis Pipeline

```powershell
python cv/analyze.py "C:\path\to\leaf.jpg" --output-dir "artifacts\reports\analysis-demo" --output-json "artifacts\reports\analysis-demo\analysis.json" --crop-hint tomato
```

This full pipeline now demonstrates:

- preprocessing and image resizing with multiple interpolation methods
- enhancement with contrast stretching, gamma correction, and unsharp masking
- restoration with median filtering and Wiener filtering
- color analysis using excess-green and lesion-score maps
- leaf-first ROI localization (SAM object segmentation when available; classical colour mask fallback) to isolate the leaf even on green-on-green field backgrounds
- deployed lesion segmentation using thresholding, clustering, and graph cut, combined with a 2-of-3 consensus (spatial-tolerance vote + healthy-tissue colour gate)
- feature extraction using color, texture, shape, and lesion-coverage features
- healthy vs diseased classification
- disease severity estimation
- infected-area percentage
- intermediate visual outputs for each major step
- baseline-vs-improved comparison outputs

Supplementary analysis outputs are still exported for comparison and presentation:

- FFT magnitude and frequency reconstructions
- Haar wavelet panel
- region growing, split-and-merge, and superpixel segmentation views

## Key Output Files

- `ml/plant/artifacts/dataset_manifest.csv`
- `ml/plant/artifacts/checkpoints/<run_name>.pt`
- `ml/plant/artifacts/production/best_model.pt`
- `ml/plant/artifacts/production/selected_model.json`
- `ml/plant/artifacts/reports/training_summary.json`
- `ml/plant/artifacts/reports/controlled_test_metrics.json`
- `ml/plant/artifacts/reports/field_test_metrics.json`
- `ml/experiments/plant/latest_leaderboard.json`

## Folder Layout

- `cv/`: computer-vision and DIP workflows
- `dl/`: deep-learning workflows
- `dl/classifier/`: internal classifier code
- `llm/`: future explanation layer scaffold
- `artifacts/`: checkpoints, reports, and manifests
- `cv/analysis_pipeline.py`: full computer-vision workflow with intermediate outputs
- `cv/analyze.py`: CLI entrypoint for the full CV pipeline
- `cv/segmentation.py`: plant-image segmentation helpers
- `cv/features.py`: handcrafted feature extraction helpers
- `dl/evaluate.py`: lightweight evaluation entrypoint

## Recommended Next Upgrade

1. Install a CUDA-enabled PyTorch build in `ml/plant/.venv` so your RTX 3050 Ti is actually used.
2. Run the `rtx3050ti_high_accuracy.json` sweep locally.
3. If local VRAM becomes a bottleneck, run the `cloud_high_accuracy.json` sweep on a larger GPU.
4. Add your own phone-camera images and re-run the sweep.

## Accuracy Expectation

- `99%+` can be realistic on cleaner controlled benchmark data.
- `99%+` on real field images is much harder and should be treated as a stretch goal, not an assumption.
- For model selection in this project, prefer `field macro F1` first, then `field accuracy`, then controlled-set metrics.
