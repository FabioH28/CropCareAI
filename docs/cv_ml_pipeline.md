# CV And ML Pipeline

## Core Idea

CropCare AI does not use only deep learning and it does not use only classical CV. It uses both together.

The plant image first goes through a practical computer vision pipeline. After that, the selected deep learning model classifies the crop and disease. Then the backend merges the structured output into a final diagnosis and explanation.

## End-To-End Pipeline

1. Load the plant image.
2. Resize it to a standard working size.
3. Run preprocessing, enhancement, and restoration.
4. Run color analysis and leaf-first ROI isolation.
5. Segment the diseased region with the core deployed segmentation methods.
6. Extract lesion and leaf features.
7. Estimate infected area percentage and severity clues.
8. Run the production CNN classifier.
9. Merge CV evidence and model prediction.
10. Generate short human-readable advice.

## What Counts As The Computer Vision Part

The CV part is everything before and around the classifier that prepares, measures, isolates, and interprets the image.

Main responsibilities:

- preprocessing
- enhancement
- restoration
- color analysis
- ROI isolation
- segmentation
- feature extraction
- infected-area estimation
- severity estimation
- intermediate visual outputs

Main files:

- `ml/plant/cv/analysis_pipeline.py`
- `ml/plant/cv/segmentation.py`
- `ml/plant/cv/features.py`

## CV Topics Actually Used

The project now keeps only the topics that are useful for the plant task as the main deployed path.

### Core deployed topics

1. basics of digital image processing
2. basic image manipulations
3. spatial and resolution changes with interpolation and resizing
4. intensity transformations and spatial filtering
5. image restoration and reconstruction
6. color image processing
7. segmentation for leaf and lesion isolation
8. feature extraction and image pattern classification
9. CV integrated with AI for smart farming
10. full end-to-end computer vision pipeline demonstration

### Supplementary comparison topics

These are still kept in the project, but not treated as the main deployed decision path:

- Fourier / frequency-domain views
- wavelet outputs
- region growing
- split-and-merge
- superpixels

## Deployed CV Path

The real deployed CV path is:

- image loading and resizing
- grayscale and RGB analysis preparation
- contrast stretching
- gamma correction
- unsharp masking
- median filtering
- Wiener filtering
- excess-green and lesion-score maps
- leaf-first ROI isolation (SAM object segmentation when available, classical excess-green/colour mask as fallback — colour alone cannot separate a green leaf from a green background, so SAM isolates the leaf by shape)
- lesion segmentation with:
  - thresholding
  - clustering
  - graph cut
- 2-of-3 consensus lesion mask, voted with a small spatial tolerance (a lesion's core and halo agree) and filtered by a healthy-tissue colour gate (a pixel counts as a lesion only if it is off the leaf's own healthy-green tone or warm-hued)
- handcrafted feature extraction
- infected-area percentage
- severity and urgency estimation

## Recent Robustness Additions

The deployed CV path was hardened after testing on real field photos:

- **SAM-assisted leaf isolation.** Colour-based masking cannot separate a green leaf from green foliage/fruit behind it. The pipeline now isolates the leaf with Segment Anything (object-shape segmentation), falling back to the classical mask if SAM is unavailable (`CROPCARE_USE_SAM=0`). Lesion analysis stays fully classical and from scratch.
- **Healthy-tissue gate + spatial-tolerance consensus.** The 2-of-3 vote is dilated slightly so a lesion's core and halo count as agreement (recall on many-spot diseases such as rust), and each consensus pixel must be chromatically off-green or warm-hued (precision: healthy leaves no longer read as diseased).
- **Quantitative evaluation.** Field accuracy is reported with 95% confidence intervals and a calibration measurement (ECE = 0.207, the model is overconfident under domain shift), and the classical leaf segmentation is benchmarked against SAM (mean IoU 0.67). See the root `README.md` and `ml/plant/artifacts/reports/`.

## Why Not Every Textbook Topic

The project does not force every CV topic from the book into the deployed path.

Reason:

- some methods genuinely improve the diagnosis path
- some are useful mainly for comparison or interpretability
- forcing every chapter equally into the final system makes the project look less coherent

So the rule is:

- core methods drive the real diagnosis path
- supplementary methods remain as extra evidence and academic comparison

## What Counts As The ML Part

The ML part is the trained deep learning classifier that predicts the crop and disease label from the prepared image.

Main files:

- `ml/plant/dl/train.py`
- `ml/plant/dl/infer.py`
- `ml/plant/dl/classifier/trainer.py`

## Selected Production Model

The current production model is:

- `tf_efficientnetv2_s`

Why it was selected:

- it won the strongest formal comparison round
- it achieved the best main field-generalization result among the promoted candidates
- it outperformed the earlier round winner and the field-first experiment winner on the chosen production selection metric

## Supported Crop And Disease Classes

The current production classifier predicts disease names only for the supported crop set below.

Overall label coverage:

- `27` total labels
- `21` diseased labels
- `6` healthy labels

Supported crop and class list:

- Apple: `healthy`, `apple_scab`, `black_rot`, `cedar_apple_rust`
- Corn: `healthy`, `common_rust`, `gray_leaf_spot`, `northern_leaf_blight`
- Grape: `healthy`, `black_rot`, `esca_black_measles`, `leaf_blight`
- Pepper: `healthy`, `bacterial_spot`
- Potato: `healthy`, `early_blight`, `late_blight`
- Tomato: `healthy`, `bacterial_spot`, `early_blight`, `late_blight`, `leaf_mold`, `septoria_leaf_spot`, `spider_mites`, `target_spot`, `mosaic_virus`, `yellow_leaf_curl_virus`

What this means in practice:

- the classifier predicts crop type and disease type for those supported classes
- `healthy` is also a real class for each supported crop family
- unsupported crops such as out-of-scope leaves are now rejected as `unknown_leaf_crop`
- unsupported crops do not receive a made-up disease name

## How CV And ML Connect

The connection is:

1. CV prepares the image and extracts measurable evidence.
2. The deep model predicts the class label.
3. The backend fuses both into one structured result.
4. The explanation layer turns that into short user-facing advice.

This means the project story is:

- CV makes the image pipeline interpretable and measurable
- ML makes the disease classification trainable and competitive
- the backend combines both into a usable app feature
