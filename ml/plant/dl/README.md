# Plant DL Layer

This folder contains the plant deep-learning classifier workflows.

Use this folder for:

- dataset manifest preparation
- model training
- checkpointed inference
- production model promotion
- evaluation

Main entrypoints:

- `prepare_dataset.py`
- `train.py`
- `infer.py`
- `evaluate.py`

Production model location:

- `ml/plant/artifacts/production/best_model.pt`

Experiment sweep:

- `ml/experiments/plant/run_model_sweep.py`
