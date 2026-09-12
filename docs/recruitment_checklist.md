# GCSRM 2026 Option B — Recruitment Checklist

This checklist maps the recruitment brief to concrete implementation locations. It is intentionally evidence-based: a row is marked complete only after the corresponding file is present and has been reviewed. Generated metrics, plots, and latency values are not treated as complete until real data has been collected and the pipeline has been run.

Legend: [Complete] implemented and reviewed · [Generated] generated after local experiment · [Pending] pending final repository audit

| Requirement | Status | Implementation location / evidence |
| --- | :---: | --- |
| Detect one hand with MediaPipe | ⏳ | `src/landmarks.py`, `deployment/app.py` |
| Extract all 21 landmarks and preserve x, y, z | ⏳ | `src/landmarks.py`, `src/features.py` |
| Gracefully handle frames without a hand | ⏳ | `src/collect_data.py`, `deployment/app.py` |
| Draw landmarks and a bounding box in deployment | ⏳ | `deployment/app.py` |
| Dedicated webcam data collector | ⏳ | `src/collect_data.py` |
| Choose gesture class and display collection progress | ⏳ | `src/collect_data.py` |
| Save valid samples only, with separate session support | ⏳ | `src/collect_data.py` |
| Retain label, sample/session/subject IDs, timestamp, and 63 coordinates | ⏳ | `src/config.py`, `src/collect_data.py`, `src/preprocess.py` |
| Explain deliberately distinct independent test session | ✅ | `README.md` → “Data collection” |
| Document 63-dimensional raw feature ordering | ✅ | `README.md` → “CSV schema and raw-coordinate ordering”; `src/features.py` |
| Engineer five normalized distances | ⏳ | `src/features.py` |
| Engineer thumb, index, and middle angles | ⏳ | `src/features.py` |
| Use wrist translation reference and middle-MCP scale | ⏳ | `src/features.py`; mathematical explanation in `README.md` |
| Guard zero / near-zero scale | ⏳ | `src/features.py`, `tests/test_features.py` |
| Keep invariant features distinct from raw coordinates | ⏳ | `src/features.py`, `notebooks/02_feature_engineering.ipynb` |
| Reject missing/unsupported labels | ⏳ | `src/config.py`, `src/preprocess.py` |
| Report class counts, percentages, and imbalance ratio | ⏳ | `src/preprocess.py`, `src/evaluate.py`, `notebooks/03_model_evaluation.ipynb` |
| Train Logistic Regression | ⏳ | `src/train.py` |
| Train Random Forest | ⏳ | `src/train.py` |
| Train RBF SVM | ⏳ | `src/train.py` |
| Compare models without selecting by cross-session test | ⏳ | `src/train.py`, `src/evaluate.py` |
| Use pipelines / training-only scaler fitting | ⏳ | `src/train.py` |
| Persist trained model, preprocessing, labels, and metadata with joblib | ⏳ | `src/train.py` |
| Avoid naive global random split | ⏳ | `src/preprocess.py`, `src/train.py` |
| Support explicit session-aware train/validation/test splits | ⏳ | `src/config.py`, `src/preprocess.py`, `src/evaluate.py` |
| Report same-session performance | ⚙️ | `src/evaluate.py` → generated `results/*_same_session.csv` |
| Report cross-session performance from a distinct session | ⚙️ | `src/evaluate.py` → generated `results/*_cross_session.csv` |
| Optionally support subject-independent evaluation | ⏳ | `src/preprocess.py`, `src/evaluate.py` |
| Required raw/invariant × same/cross generalization table | ⚙️ | `src/evaluate.py` → `results/generalization_results.csv`; evaluation notebook |
| Use actual measured values only | ✅ | `README.md`, `src/evaluate.py`, `src/benchmark.py` |
| Measure landmark-detection latency repeatedly | ⚙️ | `src/benchmark.py` → `results/latency_results.csv` |
| Measure feature-extraction latency repeatedly | ⚙️ | `src/benchmark.py` → `results/latency_results.csv` |
| Measure classifier latency separately | ⚙️ | `src/benchmark.py` → `results/latency_results.csv` |
| Measure visualization/overall-loop latency and FPS | ⚙️ | `src/benchmark.py` → `results/latency_results.csv` |
| Explain classifier vs complete-pipeline latency | ✅ | `README.md` → “Latency benchmarking” |
| Load a trained model in a live webcam app | ⏳ | `deployment/app.py` |
| Display prediction, confidence, FPS, landmarks, and bounding box | ⏳ | `deployment/app.py` |
| Exit live app cleanly | ⏳ | `deployment/app.py` |
| Deployment-only configurable temporal smoothing | ⏳ | `deployment/app.py`, `src/config.py` |
| Configurable low-confidence `UNKNOWN` state | ⏳ | `deployment/app.py`, `src/config.py` |
| Keep smoothing out of offline evaluation | ⏳ | `src/evaluate.py`, `README.md` |
| Keep data directories but prevent large data uploads | ✅ | `.gitignore`, `data/raw/.gitkeep`, `data/processed/.gitkeep` |
| Create data-generation instructions instead of fake data | ✅ | `README.md`, `notebooks/01_data_collection.ipynb` |
| Generate class-distribution plot | ⚙️ | `src/evaluate.py` → `results/class_distribution.png` |
| Generate raw and invariant confusion matrices | ⚙️ | `src/evaluate.py` → `results/confusion_matrix_*.png` |
| Generate same-session vs cross-session comparison visual | ⚙️ | `src/evaluate.py` → `results/generalization_comparison.png` |
| Summarize best model, metrics, gaps, confusions, and latency from actual results | ⚙️ | `src/evaluate.py` → generated result summary |
| Data-collection notebook explains workflow, landmarks, and schema | ✅ | `notebooks/01_data_collection.ipynb` |
| Feature notebook displays raw coordinates and normalization rationale | ✅ | `notebooks/02_feature_engineering.ipynb` |
| Evaluation notebook checks sessions, balance, models, table, and matrices | ✅ | `notebooks/03_model_evaluation.ipynb` |
| Notebook outputs do not manufacture experimental results | ✅ | all three notebooks |
| Explain leakage risks and safeguards | ✅ | `README.md` → “Leakage prevention” |
| Pin compatible dependencies and state Python version | ✅ | `requirements.txt`, `README.md` |
| Centralize paths, labels, parameters, threshold, camera, smoothing, and benchmark settings | ⏳ | `src/config.py` |
| Use modular, readable, type-oriented code with clear failures | ⏳ | source-module review |
| Test feature dimensions, malformed landmarks, zero scale, and labels | ⏳ | `tests/test_features.py`, `tests/test_preprocessing.py` |
| Use an MIT license | ✅ | `LICENSE` |

## Manual actions before submitting

- [ ] Collect diverse training, validation, and independent cross-session sessions for every gesture.
- [ ] Run training; inspect validation-only model selection and saved artifact metadata.
- [ ] Run evaluation once with the held-out cross-session session; do not tune from that result.
- [ ] Run latency benchmarking on the intended demonstration hardware.
- [ ] Review generated CSVs/figures and populate the README’s Key Findings only with measured values.
- [ ] Record the exact environment, command line, session IDs, and Git commit for the final demonstration.
- [ ] Obtain consent before sharing any identifiable webcam recordings or datasets.
