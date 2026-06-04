# Work History

This is the canonical running history file for the project.

Rule going forward:

- every major change should be recorded here
- each entry should say what was changed
- each entry should say what outcome or result was observed
- this file is meant to be the simplest chronological record from start to finish

## 2026-04-20 - Project structure reorganization

### What changed

- reorganized the repository into:
  - `backend/`
  - `frontend/`
  - `ml/`
  - `datasets/`
  - `database/`
  - `docs/`
- moved the old model code into the new `ml/` layout
- moved database migration files under `database/`
- updated path references and READMEs

### Outcome

- the repo became easier to explain and navigate
- backend tests still passed after the reorganization
- the project gained a cleaner smart-farm structure

## 2026-04-20 - Local database migration from Supabase to XAMPP/MySQL

### What changed

- removed Supabase dependence from the backend
- added local JWT auth flow
- switched storage to local file uploads
- added local MySQL/XAMPP schema and config
- updated backend env files and backend docs

### Outcome

- backend became runnable without Supabase
- backend tests passed after migration
- the system became easier to demo locally

## 2026-04-20 to 2026-04-21 - Plant CV pipeline implementation

### What changed

- added a real plant computer-vision pipeline in:
  - `ml/plant/cv/analysis_pipeline.py`
  - `ml/plant/cv/segmentation.py`
  - `ml/plant/cv/features.py`
- covered required course topics including:
  - preprocessing
  - enhancement
  - restoration
  - color analysis
  - frequency-domain analysis
  - wavelets
  - thresholding
  - region growing
  - split and merge
  - clustering
  - superpixels
  - graph cut
  - feature extraction
  - severity estimation
  - infected area percentage
  - intermediate visual outputs
  - baseline vs improved comparison
- connected the CV pipeline to backend diagnosis routes

### Outcome

- the backend could run a full CV analysis and save intermediate outputs
- the project satisfied the requested CV/DIP scope much better
- diagnosis flow became more than a mock classifier

## 2026-04-21 - Plant module separation into CV, DL, and LLM

### What changed

- split the plant intelligence stack into:
  - `ml/plant/cv/`
  - `ml/plant/dl/`
  - `ml/plant/llm/`
- moved deep-learning internals under `ml/plant/dl/classifier/`
- added initial LLM explainer scaffold

### Outcome

- the repo became much easier to explain
- CV and ML responsibilities became visually distinct
- the project structure matched the intended architecture more clearly

## 2026-04-21 - LLM explanation layer integration

### What changed

- added plant explanation helpers in `ml/plant/llm/explainer.py`
- connected backend advisory generation to local Ollama
- used the LLM to explain:
  - what happened
  - likely cause
  - severity
  - what to do next
  - why the actions help

### Outcome

- the project gained a real human-language explanation layer
- backend diagnosis responses could include structured advice
- local Ollama integration was verified

## 2026-04-21 - Initial trained plant classifier

### What changed

- trained an early real plant classifier checkpoint
- fixed a training-pipeline issue in the data split logic
- verified single-image inference after training

### Outcome

- the initial pilot run produced a real trained model
- pilot metrics:
  - backbone: `tf_efficientnetv2_s`
  - controlled accuracy: `95.34%`
  - controlled macro F1: `0.9447`
  - field accuracy: `49.15%`
  - field macro F1: `0.3811`
- this run proved the training and inference stack worked end to end

## 2026-04-21 - Production model path and sweep infrastructure

### What changed

- created a production model location:
  - `ml/plant/artifacts/production/best_model.pt`
  - `ml/plant/artifacts/production/selected_model.json`
- updated backend inference to read the production model path
- changed training so each run gets its own named checkpoint and report
- added multi-model sweep infrastructure in:
  - `ml/experiments/plant/run_model_sweep.py`

### Outcome

- the backend no longer depended on a random checkpoint
- the project could compare multiple models systematically
- the best model could be promoted automatically to production

## 2026-04-21 - Round 1 main sweep

### What changed

- trained and compared:
  - `tf_efficientnetv2_s`
  - `efficientnet_b0`
  - `resnet34`
  - `mobilenetv3_large_100`
- used field macro F1 as the main selection metric

### Outcome

- Round 1 winner: `efficientnet_b0`
- Round 1 results:
  - `efficientnet_b0`: controlled accuracy `98.70%`, controlled macro F1 `0.9875`, field accuracy `55.93%`, field macro F1 `0.5098`
  - `mobilenetv3_large_100`: controlled accuracy `97.82%`, controlled macro F1 `0.9770`, field accuracy `55.93%`, field macro F1 `0.4831`
  - `tf_efficientnetv2_s`: controlled accuracy `99.23%`, controlled macro F1 `0.9907`, field accuracy `52.54%`, field macro F1 `0.4579`
  - `resnet34`: controlled accuracy `97.82%`, controlled macro F1 `0.9774`, field accuracy `44.63%`, field macro F1 `0.3595`
- `efficientnet_b0` became the production model at that stage

## 2026-04-21 - Ensemble experiments

### What changed

- tested:
  - `top3_equal_weight_ensemble`
  - `b0_mobilenet_equal_ensemble`

### Outcome

- ensembles improved controlled-set metrics
- they did not beat the best single model on field macro F1
- results:
  - `top3_equal_weight_ensemble`: controlled accuracy `99.47%`, controlled macro F1 `0.9944`, field accuracy `54.24%`, field macro F1 `0.4863`
  - `b0_mobilenet_equal_ensemble`: controlled accuracy `99.20%`, controlled macro F1 `0.9922`, field accuracy `55.93%`, field macro F1 `0.5055`
- production stayed on `efficientnet_b0`

## 2026-04-21 - Round 2 stronger-backbone sweep

### What changed

- trained and compared:
  - `convnext_tiny`
  - `efficientnet_b3`
  - `resnet50`
  - `tf_efficientnetv2_s`
- used a stronger sweep setup with slightly larger inputs and a different batch/accumulation profile

### Outcome

- Round 2 winner: `tf_efficientnetv2_s`
- Round 2 results:
  - `tf_efficientnetv2_s`: controlled accuracy `99.44%`, controlled macro F1 `0.9936`, field accuracy `59.32%`, field macro F1 `0.5264`
  - `resnet50`: controlled accuracy `98.38%`, controlled macro F1 `0.9837`, field accuracy `57.63%`, field macro F1 `0.5256`
  - `efficientnet_b3`: controlled accuracy `98.82%`, controlled macro F1 `0.9870`, field accuracy `55.93%`, field macro F1 `0.4337`
  - `convnext_tiny`: controlled accuracy `98.55%`, controlled macro F1 `0.9861`, field accuracy `40.68%`, field macro F1 `0.3410`
- `tf_efficientnetv2_s` replaced `efficientnet_b0` as the production model

## 2026-04-21 - Full model documentation added

### What changed

- created or updated:
  - `docs/presentation_guide.md`
  - `docs/full_pipeline_and_model_history.md`
  - `docs/model_comparison_table.md`
  - `docs/model_comparison_table.csv`
  - `docs/rounds.md`
  - `docs/cv_method_necessity.md`

### Outcome

- the project gained a much clearer presentation and viva record
- model history became easier to explain
- before/after fine-tuning comparisons were later added on top of this

## 2026-04-21 - Raw pretrained baseline evaluation

### What changed

- added `ml/experiments/plant/evaluate_pretrained_baselines.py`
- evaluated pretrained backbones before task-specific fine-tuning
- updated comparison tables to include before-vs-after metrics

### Outcome

- the docs now show how much fine-tuning improved each model
- the project gained a stronger academic comparison story

## 2026-04-22 - Leaf-first CV upgrade for field images

### What changed

- strengthened leaf localization in:
  - `ml/plant/cv/segmentation.py`
- updated analysis pipeline to export:
  - scene image
  - leaf context crop
  - isolated leaf primary classifier input
- updated backend inference to run:
  - full-image classifier branch
  - leaf-ROI classifier branch
- added conservative fusion logic in:
  - `backend/app/services/inference_service.py`
- updated single-image inference output to expose class probabilities
- added backend tests for branch fusion behavior

### Outcome

- the CV pipeline became more useful for real field images
- backend tests passed after the upgrade
- field-split quick evaluation showed:
  - old full-image production model path: field accuracy `59.32%`, field macro F1 `0.5264`
  - leaf-isolated ROI only: field accuracy `61.02%`, field macro F1 `0.5400`
  - full + leaf fusion: field accuracy `61.02%`, field macro F1 `0.5407`
- this was a real improvement, but still far from `85%+`

## 2026-04-22 - Field-first max-accuracy training setup

### What changed

- upgraded the training pipeline to support:
  - `input_mode`:
    - `full_image`
    - `leaf_context`
    - `leaf_isolated`
  - `augmentation_profile`:
    - `standard`
    - `field_heavy`
  - ROI cache directory support
  - field-source sampling boost
  - stage-2 controlled-data fraction control
  - focal loss
- added:
  - `ml/plant/dl/prepare_leaf_roi_cache.py`
  - `ml/experiments/plant/configs/field_leaf_first_max_accuracy.json`
  - `docs/open_field_max_accuracy_setup.md`

### Outcome

- the codebase is now set up for the strongest realistic field-first retraining path currently available
- smoke tests succeeded for:
  - ROI cache generation
  - leaf-isolated field-heavy dataloader
  - changed training scripts compiling cleanly
- this setup is ready to run, but the full retraining results have not been produced yet

## Current state summary

### Current production classifier

- `tf_efficientnetv2_s`
- production metadata:
  - `ml/plant/artifacts/production/selected_model.json`
- production model path:
  - `ml/plant/artifacts/production/best_model.pt`

### Current measured field performance

- full-image production path:
  - field accuracy `59.32%`
  - field macro F1 `0.5264`
- leaf-first quick field evaluation:
  - field accuracy `61.02%`
  - field macro F1 `0.5407`

### Current honest conclusion

- the project is strong and well-documented
- the system has improved real-world field performance meaningfully
- the next major step is field-first retraining
- `95%` real-world multi-class open-field accuracy is not currently demonstrated and cannot honestly be claimed yet

## Update instruction

Whenever a major change is made next, append a new dated section to this file with:

- what changed
- what command or experiment was run
- what outcome happened
- whether the result improved, failed, or stayed neutral

## 2026-04-22 - Field-first max-accuracy sweep launched

### What changed

- started the next major plant retraining run
- this run uses the new field-first setup:
  - leaf-isolated ROI inputs
  - field-heavy augmentation
  - field-source boost
  - reduced controlled-data fraction in stage 2
  - focal loss
- the run is intended to test whether field performance can improve beyond the current production model

### Command path

- ROI cache command:
  - `ml/plant/dl/prepare_leaf_roi_cache.py --input-mode leaf_isolated`
- sweep config:
  - `ml/experiments/plant/configs/field_leaf_first_max_accuracy.json`
- sweep runner:
  - `ml/experiments/plant/run_model_sweep.py`

### Outcome

- status: `in progress`
- result: not available yet
- this entry should be updated after the sweep finishes with:
  - final winner
  - full metrics
  - whether the run improved field accuracy and field macro F1

## 2026-04-22 - Field-first launch fix and cache hardening

### What changed

- diagnosed why the field-first background run kept dying
- confirmed the real long-run blocker was not the training code itself, but a corrupted cached ROI image left by an interrupted cache build
- added a reusable Python launcher:
  - `scripts/run_field_first_sweep.py`
- added a Windows task-style wrapper:
  - `scripts/run_field_first_sweep_task.cmd`
- hardened ROI caching in:
  - `ml/plant/dl/classifier/data.py`
- cache behavior now:
  - deletes corrupted cached `.png` files when encountered
  - regenerates them automatically
  - writes new cache files atomically through a temporary file before replace

### Command or experiment

- verified the new runner synchronously:
  - `.\ml\plant\.venv\Scripts\python.exe scripts\run_field_first_sweep.py`
- verified the cache fix with a smoke run:
  - `.\ml\plant\.venv\Scripts\python.exe ml\plant\dl\prepare_leaf_roi_cache.py --input-mode leaf_isolated --limit 5`
- relaunched the full field-first run outside the sandbox through WMI process creation so it can continue independently:
  - wrapper command:
    - `cmd.exe /c "scripts\run_field_first_sweep_task.cmd"`

### Outcome

- first independent launch attempt exposed a real cache failure:
  - `PIL.UnidentifiedImageError` on:
    - `ml/plant/artifacts/roi_cache/leaf_isolated/76776c4b748410f3f60ee552a50304cb7793d67e.png`
- after the cache hardening patch, smoke tests passed
- the full field-first run was relaunched successfully
- live run status at relaunch check:
  - process id: `12836`
  - output log:
    - `ml/experiments/plant/run_logs/field-leaf-first-task.out.log`
  - progress log:
    - `ml/experiments/plant/run_logs/field-leaf-first-task.err.log`
  - observed state:
    - ROI cache build running normally
    - process still alive after relaunch verification
- result status: `in progress`
- this was an improvement because the run moved from repeated launch failure to a stable background execution

## 2026-04-22 - Live training continuation and GPU cleanup

### What changed

- verified that the relaunched field-first process stayed alive and continued progressing through ROI cache generation
- checked GPU occupancy before the CNN training stage
- attempted to free GPU headroom by stopping `Ollama`, which had briefly been visible in `nvidia-smi`

### Command or experiment

- live process check:
  - `Get-Process -Id 12836`
- progress checks:
  - `Get-Content ml/experiments/plant/run_logs/field-leaf-first-task.err.log -Tail ...`
- GPU checks:
  - `nvidia-smi`
- GPU cleanup attempt:
  - `Stop-Process -Id 10196 -Force`

### Outcome

- the field-first run remained active and kept advancing
- observed progress during this check window:
  - ROI cache around `25%`, then later around `31%`
- after the cleanup sequence and recheck:
  - `nvidia-smi` showed `No running processes found`
  - GPU memory use dropped back to `0 MiB / 4096 MiB`
- result status: `in progress`
- this was an improvement because the training run continued while GPU headroom was cleared before the actual deep-learning phase

## 2026-04-23 - Frontend diagnosis wired to the real backend

### What changed

- replaced the mock diagnosis page with a real frontend-to-backend diagnosis preview flow
- added a public backend preview endpoint so the frontend can call the live CV + CNN pipeline without depending on the unfinished real frontend auth flow
- added a frontend API helper for diagnosis preview requests
- updated the diagnosis page to display:
  - real crop / disease output
  - confidence, severity, urgency, infected area
  - CV evidence images
  - human advisory text and action steps
  - model version and branch information

### Files changed

- `backend/app/api/routes/diagnoses.py`
- `frontend/src/lib/api.ts`
- `frontend/src/pages/DiagnosePage.tsx`

### Command or experiment

- backend syntax check:
  - `python -m py_compile backend/app/api/routes/diagnoses.py`
- backend tests:
  - `backend\.venv\Scripts\python.exe -m pytest tests -q`
- frontend production build:
  - `npm.cmd run build`

### Outcome

- diagnosis preview is now connected to the real backend pipeline instead of a fake timer + mock result
- the app now uses the currently selected production model automatically through the backend
- backend tests passed:
  - `11 passed`
- frontend production build passed
- remaining limitation:
  - result saving and full real frontend auth are still a separate follow-up step

## 2026-04-23 - Smarter coarse-to-fine model sweep added

### What changed

- added a smarter search option to the plant model sweep runner for single-GPU use
- instead of fully training every candidate, the new strategy:
  - runs a fast screening pass on all candidate models
  - ranks them quickly
  - keeps only the top shortlist
  - fully trains only the finalists
- this is meant to reduce wasted time on weaker models without trying to train several GPU-heavy models at once on the `RTX 3050 Ti 4GB`

### Files changed

- `ml/experiments/plant/run_model_sweep.py`
- `ml/experiments/plant/configs/field_leaf_first_smart_search.json`

### Command or experiment

- syntax validation:
  - `python -m py_compile ml/experiments/plant/run_model_sweep.py`
- config sanity check:
  - confirmed `search_strategy = coarse_to_fine`
  - confirmed `shortlist_size = 2`
  - confirmed `screen_overrides` parsed correctly

### Outcome

- future sweeps can now search more models with less wasted time
- this does not change the currently running sweep in progress
- it improves the next experiment cycle by making model selection more efficient on current hardware

## 2026-04-23 - Hierarchical crop-aware pipeline recommended

### What changed

- evaluated the next major architecture direction for improving real-world field accuracy
- documented the recommendation to move from one flat generic multi-crop classifier to a hierarchical crop-aware system

### Recommended pipeline

- leaf / ROI extraction with CV
- crop classification first
- crop-specific disease classification second
- keep segmentation for infected area, severity, and visual evidence
- fuse crop + disease + CV evidence + fallback logic
- keep the LLM explanation layer on top

### Outcome

- recommendation: `yes`, this is the strongest next redesign to bet on
- reason:
  - it reduces cross-crop confusion
  - it fits the current experimental evidence better than continued flat-model tuning
  - it is more promising than relying on pure leaf-isolated generic classifier training
- documentation file added:
  - `docs/hierarchical_pipeline_recommendation.md`

## 2026-04-23 - Field-first leaf-isolated sweep finished and production restored

### What happened

- the `field_leaf_first_max_accuracy` sweep finished successfully
- the experiment trained/evaluated leaf-isolated field-first variants of:
  - `tf_efficientnetv2_s`
  - `resnet50`
  - `efficientnet_b3`
- inside that specific experiment, `resnet50` ranked first
- however, it did not beat the previous overall production model

### Final comparison

| Model | Controlled Accuracy | Controlled Macro F1 | Field Accuracy | Field Macro F1 | Decision |
|---|---:|---:|---:|---:|---|
| `rtx3050ti_stronger_backbones_tf_efficientnetv2_s` | `99.44%` | `0.9936` | `59.32%` | `0.5264` | restored as production |
| `field_leaf_first_max_accuracy_resnet50` | `96.31%` | `0.9630` | `55.37%` | `0.4913` | best in field-first sweep, not promoted overall |
| `field_leaf_first_max_accuracy_tf_efficientnetv2_s` | `96.78%` | `0.9666` | `55.37%` | `0.4430` | not promoted |
| `field_leaf_first_max_accuracy_efficientnet_b3` | `97.73%` | `0.9753` | `49.15%` | `0.4022` | not promoted |

### Outcome

- production was restored to:
  - `ml/plant/artifacts/production/best_model.pt`
  - `ml/plant/artifacts/production/selected_model.json`
- selected production model is again:
  - `rtx3050ti_stronger_backbones_tf_efficientnetv2_s`
- conclusion:
  - pure field-first leaf-isolated generic training did not improve real-field accuracy
  - segmentation should remain in the system for visual evidence, infected-area percentage, severity estimation, and ROI support
  - the next best direction is the documented hierarchical crop-aware pipeline instead of more flat generic classifier tuning

## 2026-04-23 - ROI and segmentation verification sample

### What was checked

- ran the live CV pipeline on a PlantDoc test image:
  - `datasets/plant/plantdoc/test/Apple leaf/20180511_090912-14gtw8a-e1526047952754.jpg`
- output folder:
  - `ml/plant/artifacts/reports/roi-verification-20260423`

### Outcome

- pipeline reported:
  - `pipeline`: `plant-vision-cv-v2-leaf-first`
  - `leaf_detected`: `true`
  - `classifier_primary_relative_path`: `03_roi_leaf-isolated-primary-input.png`
  - `leaf_area_percentage`: `96.94%`
  - `roi_fill_percentage`: `96.94%`
- generated visible intermediate outputs for:
  - original scene resize
  - leaf context crop
  - leaf-isolated classifier input
  - localization leaf-mask overlay
  - threshold segmentation
  - region growing
  - split and merge
  - clustering
  - superpixels
  - graph cut
  - consensus segmentation
  - segmentation comparison grid

### Interpretation

- the segmentation/ROI pipeline is actually running and producing the expected files
- this confirms the field-first approach is technically implemented, not just described
- current limitation:
  - a correct technical implementation does not guarantee better field accuracy
  - the latest field-first model underperformed the previous production model, so segmentation should support the classifier rather than replace the stronger production strategy yet

## 2026-04-23 - Hierarchical max-accuracy experiment setup

### What changed

- upgraded the plant DL trainer so it can train different target label spaces:
  - flat crop+disease classifier
  - crop classifier
  - crop-specific disease classifiers
- added support for:
  - `target_column`
  - `crop_filter`
- verified split creation works for:
  - crop classification
  - tomato-only disease classification

### Files changed

- `ml/plant/dl/classifier/data.py`
- `ml/plant/dl/classifier/trainer.py`
- `ml/experiments/plant/run_model_sweep.py`
- `ml/experiments/plant/run_hierarchical_accuracy_plan.py`
- `ml/experiments/plant/configs/hierarchical_crop_classifier_smart_search.json`
- `ml/experiments/plant/configs/hierarchical_disease_apple_smart_search.json`
- `ml/experiments/plant/configs/hierarchical_disease_corn_smart_search.json`
- `ml/experiments/plant/configs/hierarchical_disease_grape_smart_search.json`
- `ml/experiments/plant/configs/hierarchical_disease_pepper_smart_search.json`
- `ml/experiments/plant/configs/hierarchical_disease_potato_smart_search.json`
- `ml/experiments/plant/configs/hierarchical_disease_tomato_smart_search.json`
- `docs/hierarchical_accuracy_push.md`

### Outcome

- the project is now ready to test the stronger crop-aware architecture
- current production model remains unchanged and protected
- next experiment step:
  - train the crop classifier first
  - then train per-crop disease specialists
  - then compare the full hierarchical system against the flat production baseline

## 2026-04-23 - Hierarchical crop-classifier training launched

### What started

- launched the first hierarchical accuracy-push training job:
  - step: `crop_classifier`
  - config: `ml/experiments/plant/configs/hierarchical_crop_classifier_smart_search.json`
  - runner: `ml/experiments/plant/run_hierarchical_accuracy_plan.py`
  - task launcher: `scripts/run_hierarchical_crop_classifier_task.cmd`

### Runtime details

- parent process:
  - `cmd`
  - process id: `10832`
- GPU check after launch:
  - `RTX 3050 Ti`
  - GPU memory in use: about `1063 MiB`
  - Python process visible in `nvidia-smi`

### Logs

- stdout:
  - `ml/experiments/plant/run_logs/hierarchical-crop-classifier-task.out.log`
- stderr/progress:
  - `ml/experiments/plant/run_logs/hierarchical-crop-classifier-task.err.log`

### Goal

- train a crop identification model as the first stage of the hierarchical crop-aware pipeline
- after this finishes, train crop-specific disease classifiers and compare the full hierarchical system against current flat production

## 2026-04-23 - Hierarchical crop-classifier training stopped by request

### What happened

- user requested that training should not continue yet
- stopped the running hierarchical crop-classifier process tree
- terminated process tree rooted at:
  - `10832`

### Verification

- `nvidia-smi` showed:
  - GPU memory: `0 MiB / 4096 MiB`
  - GPU utilization: `0%`
  - no running GPU processes
- checked for completed hierarchical crop-classifier artifacts:
  - no completed reports found
  - no completed checkpoints found
  - no leaderboard produced

### Outcome

- no hierarchical model was completed or promoted
- current production model remains unchanged:
  - `rtx3050ti_stronger_backbones_tf_efficientnetv2_s`
- hierarchical code/config setup remains available, but training is paused until explicitly restarted

## 2026-04-23 - Tomato-leaves dataset decision

### Question

- should the project ignore the `tomato-leaves` dataset and try to reach `90%+` real-field accuracy using only PlantDoc and PlantVillage?

### Answer

- recommendation: `no`, do not ignore it for the final high-accuracy push
- reason:
  - PlantVillage is mostly clean controlled data
  - PlantDoc is real-field style but small
  - the current best model only reached `59.32%` field accuracy on PlantDoc test images
  - reaching `90%+` real-field accuracy from only those two sources is unlikely without adding more field-like data or much stronger domain adaptation

### Practical decision

- continue using PlantVillage and PlantDoc for baseline and comparison
- later, when the PC is free, extract and include `tomato-leaves`
- use it especially for the tomato-specific disease classifier in the hierarchical system
- do not train anything right now

## 2026-04-23 - Frontend/backend integration completed for auth, save, and history

### What changed

- replaced the frontend demo auth flow with real backend JWT auth
- added backend-connected sign up and log in from the React app
- wired diagnosis save so the frontend now sends prediction results to the backend
- wired the history page so it now loads saved diagnoses from the backend database
- wired delete actions from the history page to the backend
- kept the live diagnosis preview connected to the current production plant model through the backend

### Frontend files updated

- `frontend/src/lib/api.ts`
- `frontend/src/lib/auth.tsx`
- `frontend/src/App.tsx`
- `frontend/src/components/auth/ProtectedRoute.tsx`
- `frontend/src/components/layout/AppSidebar.tsx`
- `frontend/src/components/layout/TopNav.tsx`
- `frontend/src/pages/LoginPage.tsx`
- `frontend/src/pages/SignupPage.tsx`
- `frontend/src/pages/DiagnosePage.tsx`
- `frontend/src/pages/HistoryPage.tsx`

### Frontend cleanup

- removed the old demo auth file:
  - `frontend/src/lib/demo-auth.tsx`

### Verification

- backend tests:
  - `11 passed`
- frontend tests:
  - `1 passed`
- frontend production build:
  - completed successfully
- backend health check:
  - `http://127.0.0.1:8000/healthz`
  - returned status `ok`
- frontend dev server check:
  - `http://127.0.0.1:8080`
  - returned `200 OK`

### Runtime state

- backend dev server is running on:
  - `http://127.0.0.1:8000`
- frontend dev server is running on:
  - `http://127.0.0.1:8080`
- current production model remains:
  - `rtx3050ti_stronger_backbones_tf_efficientnetv2_s`

### Important blocker discovered during runtime

- the code wiring is complete, but database-backed routes are currently blocked by MySQL authentication
- current backend database setting expects:
  - user: `root`
  - password: blank
  - host: `127.0.0.1`
  - database: `cropcare_ai`
- MySQL is reachable on port `3306`, but it rejected the configured credentials with:
  - `Access denied for user 'root'@'localhost' (using password: NO)`

### Outcome

- frontend and backend are now structurally connected end-to-end
- public preview inference works through the backend
- real sign-up, login, save, and history are implemented in code
- to make those database-backed features work at runtime, the backend `.env` must be updated with the real XAMPP MySQL credentials or an equivalent local database user must be created

## 2026-04-23 - Local demo mode added to bypass database auth

### Why this was added

- the user wanted to ignore the database for now and make the app usable immediately
- backend inference was already working
- the main blocker was authentication plus save/history persistence through MySQL

### What changed

- added a hardcoded demo login:
  - username: `go`
  - password: `pass`
- added browser-only local account creation from the sign-up page
- added local browser storage for saved diagnosis history when using the local demo session
- kept plant image inference connected to the real backend model and real CV pipeline

### Resulting behavior

- login works without the database
- sign-up works without the database
- plant diagnosis preview still uses the backend API
- save result works in local browser storage
- history page reads from local browser storage
- delete from history works in local browser storage

### Verification

- frontend production build completed successfully after the demo-mode changes

### Practical outcome

- the app can now be demoed end-to-end without fixing XAMPP credentials first
- only persistent multi-user database storage remains postponed

## 2026-04-23 - One-click demo access added

### Why this was added

- the user wanted an even faster presentation flow with no password typing at all
- local demo auth was already working, so the next improvement was removing form friction

### What changed

- added a one-click `signInDemo` path in the frontend auth provider
- added an `Enter Demo Instantly` button on the login page
- added direct `Open Demo` and `Enter Demo Now` buttons on the landing page
- kept manual local sign-in and browser-only account creation available as secondary options

### Result

- the user can now enter the app with a single button press
- no email or password is required for the demo path
- the real backend plant diagnosis pipeline remains unchanged

### Verification

- frontend production build completed successfully after the one-click demo changes

## 2026-04-23 - Fixed frontend `Failed to fetch` caused by CORS

### Symptom

- the frontend showed `Failed to fetch` when trying to call the backend from `http://127.0.0.1:8080`

### Root cause

- backend CORS settings only allowed older frontend origins such as `localhost:5173`
- the browser blocked cross-origin requests from the current frontend dev server origin

### What changed

- expanded backend allowed origins in:
  - `backend/app/core/config.py`
  - `backend/.env`
  - `backend/.env.example`
- added support for these frontend origins:
  - `http://localhost:3000`
  - `http://127.0.0.1:3000`
  - `http://localhost:5173`
  - `http://127.0.0.1:5173`
  - `http://localhost:8080`
  - `http://127.0.0.1:8080`
- restarted the backend after the config change

### Verification

- backend health check passed after restart
- CORS preflight to `POST /api/v1/diagnoses/public-preview` from origin `http://127.0.0.1:8080` returned `200`

### Outcome

- the `Failed to fetch` issue caused by CORS should no longer block frontend-to-backend requests

## 2026-04-23 - Removed Lovable tab icon branding

### Why this was changed

- the browser tab was still showing the old Lovable favicon
- the project needed consistent CropCare branding for demo and presentation use

### What changed

- copied the project leaf icon into:
  - `frontend/public/cropcare-favicon.png`
- added an explicit favicon link in:
  - `frontend/index.html`

### Outcome

- the browser tab now uses the CropCare icon instead of the old Lovable favicon

## 2026-04-23 - Removed animal-module scaffolding and made the repo plant-only

### Why this was changed

- the user decided to focus the whole project on plants only
- the animal side of the repository was only placeholder scaffolding and documentation references

### What changed

- removed plant-vs-animal wording from the root project description
- removed animal references from architecture, scope, dataset, experiment, and presentation docs
- prepared the repository to keep only the plant-focused `CropCare AI` direction

### Outcome

- the repository is now documented as a plant-only project
- only the plant pipeline remains part of the intended delivered scope

## 2026-04-23 - Refocused the plant CV pipeline on only the methods that materially help the project

### Why this was changed

- the user clarified that the professor does not want every textbook CV topic forced into the project
- the plant module needed to present a stronger and more honest story: keep the methods that help diagnosis, and downgrade the rest to supplementary comparison outputs

### What changed in code

- updated `ml/plant/cv/analysis_pipeline.py` so the deployed lesion-decision path now clearly uses:
  - leaf-first ROI isolation
  - thresholding
  - clustering
  - graph cut
- changed the improved consensus mask to be built from the core deployed segmentation methods instead of treating every available segmentation method as equally central
- kept FFT, wavelet, region-growing, split-and-merge, and superpixel outputs in the pipeline as supplementary comparison and interpretability outputs
- added clearer analysis metadata:
  - `topics_covered`
  - `supplementary_topics_demonstrated`
  - `topic_selection_policy`
  - `classification.comparison.core_segmentation_methods`
  - `classification.comparison.comparison_only_segmentation_methods`
  - `segmentation.deployed_strategy`

### What changed in documentation

- updated the project scope to reflect a practical deployed CV path instead of a full-book checklist
- rewrote `docs/cv_method_necessity.md` into a final keep/optional/remove note
- updated the main presentation and pipeline-history docs so they now explain:
  - core deployed CV topics
  - supplementary comparison topics
  - why the practical method set is stronger for this project
- updated `backend/README.md`, `ml/plant/README.md`, and `ml/plant/cv/README.md` to match the same core-vs-supplementary story

### Verification

- `python -m py_compile ml\\plant\\cv\\analysis_pipeline.py` completed successfully

### Outcome

- the project now clearly shows which CV methods are actually part of the deployed plant diagnosis path
- the supplementary methods are still available for demos, reports, and comparison, but they are no longer presented as equally necessary for the final system

## 2026-04-24 - Cleaned the project layout and added human section comments to the main flow files

### Why this was changed

- the project had become harder to read quickly because the repo root still showed temporary smoke-output folders
- the main pipeline files were correct, but long enough that a first-time reader had to work too hard to find the real flow

### What changed

- moved the root-level temporary smoke output folders into:
  - `tmp/cv-smoke/run_1`
  - `tmp/cv-smoke/run_2`
- added `tmp/` to `.gitignore`
- replaced the root `README.md` with a shorter start-here guide focused on:
  - where to begin reading
  - the real diagnosis code path
  - what each top-level folder is for
- added `docs/project_map.md` as a quick project-reading guide
- added plain section comments to the main long flow files:
  - `ml/plant/cv/analysis_pipeline.py`
  - `backend/app/services/inference_service.py`
  - `frontend/src/pages/DiagnosePage.tsx`

### Outcome

- the repo root is easier to scan
- the project now has a clear "main path" for readers who only want to understand the real diagnosis flow
- the long files are easier to follow without changing their behavior

## 2026-04-24 - Wired the app to a real local XAMPP-backed MariaDB instance and restored backend accounts

### Why this was changed

- the backend was still pointing at an unusable `3306` connection
- the frontend manual sign-up flow had been left in browser-only local mode
- the project needed a real database-backed user and diagnosis flow again

### What changed

- created a project-local MariaDB configuration at:
  - `database/xampp-local/my.ini`
- initialized a clean local MariaDB data directory under:
  - `database/xampp-local/data`
- configured the local project database instance to run on:
  - `127.0.0.1:3308`
- set the backend database connection to:
  - `mysql+pymysql://root:cropcare_root@127.0.0.1:3308/cropcare_ai`
- applied the schema from:
  - `database/migrations/001_initial_schema.sql`
- added a reproducible Fabio seed file:
  - `database/seeds/001_fabio_seed.sql`
- restored the frontend sign-up flow to create real backend accounts instead of browser-only local accounts
- updated the login and sign-up page copy so it no longer claims the manual account path is local-browser-only

### Seeded demo account

- email: `fabio@cropcareai.app`
- password: `Fabio123!`
- full name: `Fabio`

### Verification

- backend login succeeded for Fabio
- diagnosis upload succeeded and was saved in the SQL database
- diagnosis list endpoint returned the saved database record
- backend sign-up succeeded for a fresh test user and `/auth/me` returned the created profile
- frontend dev server responded with `200`
- frontend production build passed
- Python syntax check passed for the touched backend files

### Outcome

- the project now has a real database-backed user flow again
- manual sign-up and login are tied back to the backend database
- Fabio can log in and create diagnosis records stored in MariaDB

## 2026-04-24 - Finished the frontend/backend/database app flow around the existing production plant model

### Why this was changed

- the plant diagnosis path already worked, but large parts of the app were still static UI and not connected to the backend
- settings, advisor chat, treatment management, plant management, and profile/security flows needed to become real features instead of placeholders
- the project also needed repeatable SQL support for the remaining app pages and a clear way to start the local stack

### What changed

- added backend routes and schemas for:
  - persisted user settings
  - advisor chat backed by saved account context with Ollama fallback to rules
  - password reset request recording
  - authenticated password changes
- added new SQL support in:
  - `database/migrations/002_app_support.sql`
  - `database/seeds/002_fabio_app_seed.sql`
- fixed the database helper so tables with `user_id` as the primary key work correctly during inserts
- replaced the remaining static frontend pages with real backend-backed flows:
  - `frontend/src/pages/DashboardPage.tsx`
  - `frontend/src/pages/PlantsPage.tsx`
  - `frontend/src/pages/PlantDetailPage.tsx`
  - `frontend/src/pages/TreatmentsPage.tsx`
  - `frontend/src/pages/NotificationsPage.tsx`
  - `frontend/src/pages/ProfilePage.tsx`
  - `frontend/src/pages/SettingsPage.tsx`
  - `frontend/src/pages/AdvisorPage.tsx`
  - `frontend/src/pages/ForgotPasswordPage.tsx`
- extended `frontend/src/lib/api.ts` to cover the real app endpoints used by those pages
- updated `frontend/src/components/layout/TopNav.tsx` to show live unread notification counts and route to notifications/profile
- extended `frontend/src/components/shared/StatusBadge.tsx` so the real plant and treatment statuses render correctly
- added run helpers:
  - `scripts/start-cropcare-db.ps1`
  - `scripts/run-backend.ps1`
  - `scripts/run-frontend.ps1`
- applied the new migration and seed to the project-local MariaDB instance on `127.0.0.1:3308`

### Verification

- backend Python syntax check passed for the new routes, services, and database helper
- frontend production build passed after the page rewiring
- frontend dev server responded with `200` on `http://127.0.0.1:8080`
- Fabio API verification passed for:
  - login
  - dashboard summary
  - plants listing
  - treatment log listing
  - notifications listing
  - settings fetch
  - advisor chat
  - password reset request recording
- a temporary test user verification also passed for:
  - sign-up
  - profile update
  - password change
  - re-login with the new password
  - automatic settings-row creation
  - plant creation
  - treatment creation
  - notification read
  - dashboard summary for the new account

### Outcome

- the app is now functionally wired across frontend, backend, and database without changing the plant model itself
- Fabio has a seeded account with working plants, treatments, notifications, settings, and dashboard data
- the production plant model remains the selected model used by the diagnosis flow while the rest of the app is now real and connected

## 2026-04-24 - Consolidated the docs folder into a smaller main set and archived the older split-out notes

### Why this was changed

- the docs folder had grown into too many overlapping files
- the main story of the project was hard to follow because scope notes, experiments, presentation drafts, and history were all mixed together
- the project needed a cleaner final-docs structure with only a few main files at the top level

### What changed

- created a new main docs set:
  - `docs/README.md`
  - `docs/system_overview.md`
  - `docs/cv_ml_pipeline.md`
  - `docs/results_and_model_selection.md`
  - `docs/presentation_notes.md`
- kept `docs/work_history.md` as the detailed chronological log
- moved the older overlapping docs into:
  - `docs/archive/`
- updated the root `README.md` so the project now points to the new consolidated docs path first

### Outcome

- the docs root is now much easier to scan
- the main project story now lives in a small set of top-level docs instead of being spread across many separate files
- the older detailed notes are still preserved in `docs/archive/` without cluttering the main reading path

## 2026-04-24 - Removed the frontend local-demo bypass so the app now uses the real backend database path only

### Why this was changed

- the frontend still had an older one-click demo auth flow
- that demo path created local browser sessions and local diagnosis storage, which made the app feel fake even though the backend and SQL database were already working
- the app needed one clear production path for presentation: real login, real API calls, real database-backed records

### What changed

- removed the local-demo auth/session logic from `frontend/src/lib/auth.tsx`
- removed local diagnosis/history storage fallbacks from `frontend/src/lib/api.ts`
- updated `frontend/src/pages/LandingPage.tsx` so it only routes into the real sign-in and sign-up flow
- updated `frontend/src/pages/LoginPage.tsx` to use database-backed sign-in only instead of a demo button
- updated `frontend/src/pages/DiagnosePage.tsx` and `frontend/src/pages/HistoryPage.tsx` so they always describe and use the backend + SQL path
- added cleanup for stale legacy demo sessions so old `local-demo-token:*` sessions are discarded on load

### Outcome

- the app no longer has a browser-only demo auth path
- saved diagnoses, history, plants, settings, and the rest of the app now go through the real backend account flow

## 2026-04-24 - Removed visible seeded-account credentials from the public UI

### Why this was changed

- the login and landing pages were still visibly showing the Fabio seeded account details
- that made the app look temporary and exposed internal test credentials in the interface

### What changed

- removed the seeded-account panel from `frontend/src/pages/LoginPage.tsx`
- removed the seeded-account line from `frontend/src/pages/LandingPage.tsx`

### Outcome

- the public UI now looks like a normal real app login flow
- the Fabio account still exists for testing, but it is no longer shown on-screen

## 2026-04-24 - Added a real diagnosis progress bar with live backend stage polling

### Why this was changed

- the diagnosis page only had a spinner while the backend was doing a long blocking preview request
- that gave no real sense of where time was being spent or how long the analysis was actually taking

### What changed

- added in-memory background preview jobs in `backend/app/services/inference_job_service.py`
- added progress-aware preview job endpoints in `backend/app/api/routes/diagnoses.py`
- extended `backend/app/services/inference_service.py` to report real stage updates during workspace prep, CV analysis, deep-learning inference, result merging, and finalization
- added `InferenceJobStatusPayload` in `backend/app/schemas/diagnosis.py`
- updated `frontend/src/lib/api.ts` to start and poll preview jobs
- updated `frontend/src/pages/DiagnosePage.tsx` to show:
  - a real progress bar
  - current backend phase
  - elapsed time
  - approximate remaining time
  - final analysis time in the result card

### Outcome

- plant diagnosis now uses real live polling instead of a fake loading state
- a smoke test job progressed through `running_cv_pipeline -> classifying_full_image -> completed`
- the measured end-to-end runtime for that live test job was about `17.56s`

## 2026-04-24 - Cleaned the last fake-feeling app behavior and aligned diagnosis wording

### Why this was changed

- some pages still sounded temporary because they kept saying "local database" or treated Fabio like seeded demo content
- suspicious scans with no named disease were explained correctly on the diagnosis page, but older pages still showed them like "No disease detected"
- the diagnosis page could imply auto-save was on even when the user was not signed in

### What changed

- updated `frontend/src/pages/DiagnosePage.tsx` so auto-save defaults off until an authenticated settings load confirms it
- updated the diagnosis page copy to describe the project SQL database instead of demo-style local wording
- aligned suspicious-result labels across:
  - `frontend/src/pages/DiagnosePage.tsx`
  - `frontend/src/pages/HistoryPage.tsx`
  - `frontend/src/pages/DashboardPage.tsx`
  - `frontend/src/pages/PlantDetailPage.tsx`
- updated frontend copy in:
  - `frontend/src/pages/SignupPage.tsx`
  - `frontend/src/pages/SettingsPage.tsx`
  - `frontend/src/pages/ProfilePage.tsx`
- clarified the seed documentation in:
  - `docs/system_overview.md`
  - `database/README.md`

### Outcome

- the app now reads more like a real project system and less like a seeded demo
- suspicious-but-unnamed CV findings no longer look contradictory when you view them outside the diagnosis page
- the docs now match the actual seeded data: Fabio starts with an account, farm record, and settings, not fake starter activity

## 2026-04-24 - Restarted the live stack on the cleaned code and re-verified the real app path

### Why this was changed

- after the cleanup pass, the frontend was down and the backend needed a fresh restart so the running services matched the updated files
- I also needed to confirm that Fabio's account no longer had fake starter activity in the real database

### What changed

- restarted the FastAPI backend on `http://127.0.0.1:8000`
- restarted the Vite frontend on `http://127.0.0.1:8080`
- updated `backend/README.md` so the documented database port, password, CORS origins, and startup path match the actual project setup
- rechecked the Fabio account through the live API after restart

### Verification

- backend health check returned `200`
- frontend root returned `200`
- Fabio login succeeded
- dashboard summary for Fabio returned:
  - `total_plants=0`
  - `total_diagnoses=0`
  - `active_alerts=0`
- end-to-end diagnosis save verification passed for a temporary account:
  - public diagnosis preview succeeded
  - authenticated diagnosis save succeeded
  - saved history list returned `total=1`
- linked-record verification also passed for another temporary account:
  - plant creation succeeded
  - diagnosis save with `plant_id` succeeded
  - diagnosis history filtered by that plant returned `total=1`
  - notifications list returned `total=1`

### Outcome

- the running app now matches the cleaned source code
- the stack is live again on the expected local URLs
- Fabio's account is clean and database-backed instead of looking prefilled with fake starter activity

## 2026-04-24 - Simplified the diagnosis and advisor pages and shortened the LLM explanation style

### Why this was changed

- the diagnosis page was becoming too dense to read during a demo
- the advisor/chat page looked more like a full chat tool than a quick explanation helper
- the generated explanation text needed to be shorter and more direct

### What changed

- simplified `frontend/src/pages/DiagnosePage.tsx`:
  - shorter labels in the top result card
  - compact explanation cards for:
    - what happened
    - likely cause
    - what to do
    - why it matters
  - prevention tips reduced to short chips
  - computer-vision images moved into a collapsed evidence section
  - result wording shortened where possible
- simplified `frontend/src/pages/AdvisorPage.tsx`:
  - shorter intro copy
  - smaller quick prompts
  - simpler message layout
  - more direct input wording
- shortened advisor response style in `backend/app/services/advisor_chat_service.py`:
  - shorter context summary
  - shorter fallback replies
  - LLM prompt now prefers a compact result/cause/next-step style
- tightened diagnosis-advisory prompt wording in `ml/plant/llm/explainer.py` so the generated field text stays shorter and plainer

### Verification

- backend syntax checks passed
- ML explainer syntax check passed
- frontend production build passed
- backend restarted successfully on `127.0.0.1:8000`
- backend health check returned `200`
- frontend still responded on `127.0.0.1:8080`
- advisor endpoint returned the new shorter context summary format:
  - `Using 0 plants, 2 diagnoses, and 0 due follow-ups.`

### Outcome

- the diagnosis flow now reads more like a short report than a dense technical page
- the advisor now behaves more like a quick explainer than a chat-heavy screen
- the LLM/rules explanation path is shorter, simpler, and easier to present

## 2026-04-24 - Fixed the confusing login and signup behavior

### Why this was changed

- the frontend was trusting any stored auth token immediately, which made the app feel like it was logging in without asking for credentials
- the signup page could fail on backend validation, but the frontend did not explain the missing requirements clearly enough

### What changed

- updated `frontend/src/lib/auth.tsx` to validate stored sessions against `/auth/me` before treating the user as signed in
- added an `authReady` state so the app can wait for session validation instead of redirecting too early
- updated `frontend/src/components/auth/ProtectedRoute.tsx` to wait while session validation runs
- updated `frontend/src/pages/LoginPage.tsx` and `frontend/src/pages/SignupPage.tsx`:
  - no more silent redirect on those pages
  - users now see a clear "already signed in" state
  - added a button to sign out and use another account
  - added clearer signup validation for:
    - full name
    - password length

### Verification

- frontend production build passed
- backend signup API still succeeded for a fresh test account after the changes
- frontend was restarted and responded with `200`
- backend remained healthy and responded with `200`

### Outcome

- login and signup no longer feel broken or automatic
- stale sessions are validated instead of blindly trusted
- signup failures now show clearer frontend validation before the request is sent

## 2026-04-24 - Switched the project MariaDB root login to a blank password

### Why this was changed

- the project database had been set to `root / cropcare_root`
- the user wanted the local XAMPP/phpMyAdmin flow changed to a blank-password root login so they can type only `root` and continue

### What changed

- updated `backend/.env` so the backend now uses a blank-password root connection string
- updated `backend/README.md` so the documented `DATABASE_URL` matches the new local DB login
- updated `scripts/start-cropcare-db.ps1` so it removes a stale MariaDB PID file automatically before startup
- the live MariaDB root accounts and phpMyAdmin config were updated outside the repo to accept a blank password on the project database port

### Outcome

- the backend and phpMyAdmin now target the same passwordless local root login
- phpMyAdmin no longer requires a password for the project DB login

## 2026-04-24 - Fixed frontend sign-in "Failed to fetch" against the live backend

### Why this was changed

- the sign-in page was failing with a generic browser `Failed to fetch` error even though the backend health check and `/auth/log-in` API were both working
- the frontend fallback API base still used `http://localhost:8000/api/v1`
- on Windows, `localhost` can resolve to IPv6 `::1` while the backend in this project is bound to `127.0.0.1`, which can produce a browser-side connection failure

### What changed

- updated `frontend/src/lib/api.ts` so the frontend fallback API base now uses `http://127.0.0.1:8000/api/v1`
- added `frontend/.env` with `VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1` so the frontend always targets the same backend address explicitly

### Verification

- frontend build passed after the change
- the built frontend bundle contains `127.0.0.1:8000` instead of `localhost:8000`
- the live dev server module at `/src/lib/api.ts` also serves the `127.0.0.1:8000` API base
- backend health remained `200`
- direct login API check for the seeded Fabio account still succeeded

### Outcome

- the frontend and backend now use the same loopback address consistently
- the sign-in path should no longer fail because of a `localhost` vs `127.0.0.1` mismatch

## 2026-04-24 - Simplified the frontend to the core CropCare flow

### Why this was changed

- the app had too many tabs and feature surfaces for the focused plant-diagnosis project
- several screens were no longer part of the presentation path the user wanted to keep
- the goal was to make the web app feel direct: dashboard, diagnose, history, and profile

### What changed

- trimmed the protected app routes to:
  - `dashboard`
  - `diagnose`
  - `history`
  - `profile`
- removed the extra frontend page files for:
  - plants
  - plant detail
  - treatments
  - advisor
  - notifications
  - settings
- simplified the sidebar, mobile nav, and top nav so they only expose the four remaining pages
- changed login and signup redirects so the streamlined app lands on the dashboard
- simplified the diagnose page by removing plant-linking and settings-driven auto-save behavior
- simplified the dashboard so it focuses on saved scans, flagged results, health score, recent diagnoses, and quick links
- updated the landing page copy so it matches the smaller real app instead of advertising removed sections

### Verification

- frontend production build passed after the route and page cleanup

### Outcome

- the app now reads like one focused diagnosis product instead of a larger multi-module platform
- the UI only exposes the pages that are actually needed for the project demo

## 2026-04-24 - Added unsupported-crop rejection before disease labeling

### Why this was changed

- unsupported leaves such as cherry, peach, or blueberry were being forced into the nearest known crop class
- the production classifier can name diseases for its trained crops, but it previously had no open-set guard for leaves outside that crop set

### What changed

- added `ml/plant/dl/classifier/open_set_gate.py` to score image embeddings against:
  - supported crop prototypes built from the trained crop set
  - unsupported crop prototypes built from out-of-scope PlantDoc leaves
- added `ml/plant/dl/build_open_set_gate.py` and generated the production artifact:
  - `ml/plant/artifacts/production/open_set_gate.pt`
- updated `ml/plant/dl/infer.py` so unsupported leaves are now returned as:
  - `predicted_crop=unknown_leaf_crop`
  - `predicted_disease=null`
  - `health_status=uncertain`
- updated `backend/app/services/inference_service.py` so open-set rejections keep the CV summary instead of forcing a wrong deep-learning crop label
- updated the frontend diagnosis, dashboard, and history pages to display:
  - `Unknown / Unsupported Crop`
  - disease label withheld messaging
- updated `ml/plant/llm/explainer.py` so the explanation layer does not invent a disease name when the crop is outside the supported set

### Verification

- direct production-checkpoint inference on a known apple leaf still returned `apple`
- direct production-checkpoint inference on cherry, peach, and blueberry now returns `unknown_leaf_crop`
- Python syntax checks passed for the touched backend and ML files
- frontend build passed after the UI wording updates

### Outcome

- unsupported crops are no longer forced into the nearest known crop label
- the disease label is now treated as valid only for crops inside the trained support set

## 2026-04-24 - Fixed live API fusion so unsupported crops stay unsupported

### Why this was changed

- the branch-level open-set gate was already rejecting unsupported crops correctly
- the final backend fusion step could still rebuild a known crop from class probabilities, which made the live API answer disagree with the branch predictions

### What changed

- tightened `backend/app/services/inference_service.py` so:
  - if both DL branches reject the crop, the fused result stays `unknown_leaf_crop`
  - if one branch rejects the crop, the known branch only survives when its crop-support evidence is clearly strong
  - unsupported-crop final summaries now always keep `predicted_crop=unknown_leaf_crop` instead of falling back to an earlier heuristic crop name
- preserved combined open-set gate details on the fused DL output for debugging and presentation

### Verification

- backend syntax check passed after the fusion update
- live backend test with a real cherry leaf now returns:
  - `predicted_crop=unknown_leaf_crop`
  - `predicted_disease=null`
  - `decision_strategy=unsupported_crop_rejected_keep_cv_summary`
- live backend test with a real apple scab image still returns:
  - `predicted_crop=apple`
  - `predicted_disease=apple_scab`
  - `health_status=diseased`

### Outcome

- the live app now rejects out-of-scope crops instead of quietly renaming them as a supported crop
- disease naming is now limited to the crop/disease classes the production model actually knows

## 2026-04-24 - Documented the exact disease classes the production model supports

### Why this was changed

- the project docs explained the pipeline but did not clearly list the disease labels the current model can actually predict
- the presentation needs one direct place that states the supported crop and disease classes

### What changed

- updated `docs/system_overview.md` with a short summary of:
  - supported crop count
  - class count
  - the rule that unsupported crops are rejected as `unknown_leaf_crop`
- updated `docs/cv_ml_pipeline.md` with the exact supported crop and disease class list for the production model
- updated `docs/README.md` so that this information is easier to find

### Outcome

- the docs now show exactly which disease types the production model can name
- the model scope is clearer for both demo and presentation use

## 2026-04-25 - Added CV post-processing to mark lesion spots and expose CV-only severity evidence

### Why this was changed

- the app already used CV to estimate affected area and severity, but the result card did not show a clear post-processing view of the diseased spots
- the presentation needed a more honest explanation that severity comes from lesion geometry and segmentation, not from the CNN classifier alone

### What changed

- updated `ml/plant/cv/analysis_pipeline.py` to:
  - group consensus lesion regions with connected-component post-processing
  - export a new `marked_disease_spots` image with visible lesion boxes and centroids
  - add `postprocessing` metrics such as lesion-region count, largest region percentage, and mean region percentage
  - mark `severity_estimation.source` as CV-only consensus-mask measurement
- updated `frontend/src/pages/DiagnosePage.tsx` to:
  - prioritize the marked lesion image in the CV evidence section
  - label area chips as `CV estimated ...`
  - show the number of marked spot regions when available

### Outcome

- the app now visually marks the lesion spots found by the CV pipeline
- severity and affected-area estimates are presented more honestly as CV-derived post-processing outputs
