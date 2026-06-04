# Plant LLM Layer

This folder is reserved for the language-model layer that explains plant diagnosis outputs.

Recommended role:

- take the structured output from `ml/plant/cv/analysis_pipeline.py`
- optionally combine it with the trained classifier output from `ml/plant/dl/`
- generate farmer-friendly explanations, next-step advice, and report summaries
- produce short sections such as what happened, likely cause, severity, what to do, and why it helps

Current local setup:

- backend uses a local Ollama model by default
- default model: `llama3:8b`
- main helper file: `explainer.py`

Suggested future files:

- `prompt_templates.py`
- `explanation_service.py`
- `report_generator.py`
- `guardrails.py`
