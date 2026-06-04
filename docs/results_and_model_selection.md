# Results And Model Selection

## Selection Rule

The project does not choose the winning model only by clean controlled-image accuracy.

Priority:

1. field macro F1
2. field accuracy
3. controlled macro F1
4. controlled accuracy

This was done because the real target is practical field performance, not only laboratory-style image performance.

## Pilot Run

Before the formal rounds, there was one earlier real run used to prove the training pipeline worked end-to-end.

- model: `tf_efficientnetv2_s`
- controlled accuracy: `95.34%`
- controlled macro F1: `0.9447`
- field accuracy: `49.15%`
- field macro F1: `0.3811`

## Round 1

Goal:

- compare strong, practical backbones that could train on the local RTX 3050 Ti 4GB machine

Candidates:

- `efficientnet_b0`
- `mobilenetv3_large_100`
- `tf_efficientnetv2_s`
- `resnet34`

### Round 1 results

| Model | Controlled Accuracy | Controlled Macro F1 | Field Accuracy | Field Macro F1 | Outcome |
|---|---:|---:|---:|---:|---|
| `efficientnet_b0` | 98.70% | 0.9875 | 55.93% | 0.5098 | Round 1 winner |
| `mobilenetv3_large_100` | 97.82% | 0.9770 | 55.93% | 0.4831 | Competitive but below winner |
| `tf_efficientnetv2_s` | 99.23% | 0.9907 | 52.54% | 0.4579 | Strong clean-set result, weaker field result |
| `resnet34` | 97.82% | 0.9774 | 44.63% | 0.3595 | Useful baseline only |

Round 1 winner:

- `efficientnet_b0`

## Round 2

Goal:

- push for a stronger final production model after Round 1

Candidates:

- `tf_efficientnetv2_s`
- `resnet50`
- `efficientnet_b3`
- `convnext_tiny`

### Round 2 results

| Model | Controlled Accuracy | Controlled Macro F1 | Field Accuracy | Field Macro F1 | Outcome |
|---|---:|---:|---:|---:|---|
| `tf_efficientnetv2_s` | 99.44% | 0.9936 | 59.32% | 0.5264 | Round 2 winner and current production |
| `resnet50` | 98.38% | 0.9837 | 57.63% | 0.5256 | Very close, but not the winner |
| `efficientnet_b3` | 98.82% | 0.9870 | 55.93% | 0.4337 | Not promoted |
| `convnext_tiny` | 98.55% | 0.9861 | 40.68% | 0.3410 | Not promoted |

Round 2 winner:

- `tf_efficientnetv2_s`

## Ensemble Tests

After Round 1, ensembles were tested to see whether combining models beat the best single model.

Results:

- they improved some controlled-set metrics
- they did not beat the best single model on the main field metric

So they were not promoted.

## Field-First Leaf-Isolated Experiment

This experiment tested whether heavier leaf isolation and field-first preparation alone would beat the strongest standard production path.

Best result from that experiment:

- winner: `resnet50`
- controlled accuracy: `96.31%`
- controlled macro F1: `0.9630`
- field accuracy: `55.37%`
- field macro F1: `0.4913`

Conclusion:

- useful experiment
- helpful for CV realism
- not good enough to replace the stronger Round 2 production model

## Final Production Model

Current production model:

- `tf_efficientnetv2_s`

Why it stayed in production:

- best main field metric among promoted candidates
- stronger than the Round 1 winner
- stronger than the field-first experiment winner
- balanced clean performance and field generalization

## Main Takeaway

The project history shows:

- pretrained backbones alone were not enough
- task-specific fine-tuning gave the large performance jump
- model comparison mattered more than picking one architecture blindly
- segmentation is helpful for ROI and interpretability, but alone it did not beat the best selected CNN pipeline
