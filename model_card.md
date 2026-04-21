# Model Card: Chronic Kidney Disease (CKD) Classifier

## Model Details

| Field | Value |
|---|---|
| **Model name** | `kidney-disease-baseline` |
| **Algorithm** | LightGBM (`LGBMClassifier`) |
| **MLflow Registry alias** | `Production` |
| **Experiment** | `ckd-cloud-experiment` |
| **Serving endpoint** | Cloud Run — `prediction-api` (europe-west1) |
| **Framework** | scikit-learn Pipeline (preprocessing + LightGBM) |
| **Python version** | 3.10 |

---

## Intended Use

This model is designed as a **CKD screening aid** for clinical informatics pipelines. Given routine lab measurements and patient observations, it outputs a binary prediction:

- `0` — CKD **not detected**
- `1` — CKD **detected**

**Not intended for:** autonomous clinical diagnosis, replacement of physician judgement, or deployment in populations not represented by the UCI CKD dataset.

---

## Training Data

- **Source:** UCI Machine Learning Repository — Chronic Kidney Disease dataset
- **Storage:** `gs://chronic-kd-kidney-disease-ml/raw_data/kidney_disease_train.csv`
- **Size:** ~400 patient records
- **Train/test split:** 80% / 20% (stratified, `random_state=42`)
- **Target variable:** `classification` (binary: `ckd` → 1, `notckd` → 0)

---

## Features

### Numeric Features (11)

| Feature | Description |
|---|---|
| `age` | Age in years |
| `bp` | Blood pressure (mm/Hg) |
| `bgr` | Blood glucose random (mgs/dl) |
| `bu` | Blood urea (mgs/dl) |
| `sc` | Serum creatinine (mgs/dl) |
| `sod` | Sodium (mEq/L) |
| `pot` | Potassium (mEq/L) |
| `hemo` | Haemoglobin (gms) |
| `pcv` | Packed cell volume |
| `wc` | White blood cell count (cells/cumm) |
| `rc` | Red blood cell count (millions/cmm) |

### Categorical Features (13)

| Feature | Description | Values |
|---|---|---|
| `sg` | Specific gravity | 1.005, 1.010, 1.015, 1.020, 1.025 |
| `al` | Albumin | 0–5 |
| `su` | Sugar | 0–5 |
| `rbc` | Red blood cells | normal, abnormal |
| `pc` | Pus cell | normal, abnormal |
| `pcc` | Pus cell clumps | present, notpresent |
| `ba` | Bacteria | present, notpresent |
| `htn` | Hypertension | yes, no |
| `dm` | Diabetes mellitus | yes, no |
| `cad` | Coronary artery disease | yes, no |
| `appet` | Appetite | good, poor |
| `pe` | Pedal oedema | yes, no |
| `ane` | Anaemia | yes, no |

All features are optional — missing values are imputed by the preprocessing pipeline (median for numeric, most-frequent for categorical).

---

## Preprocessing Pipeline

Built with `sklearn.pipeline.Pipeline`:

1. **Numeric:** `SimpleImputer(strategy='median')` → `StandardScaler`
2. **Categorical:** `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`

---

## Performance (Test Set — 20% held-out)

All three models were trained and evaluated. The winner is promoted to Production only if it beats the best existing Production run.

| Model | F1 (macro) | Precision | Recall | Accuracy |
|---|---|---|---|---|
| **LightGBM** ✅ | **0.964** | **0.965** | **0.964** | **0.966** |
| XGBoost | 0.964 | 0.965 | 0.964 | 0.966 |
| RandomForest | 0.964 | 0.965 | 0.964 | 0.966 |

*LightGBM was selected as Production model (tie broken by training order).*

### Error Analysis

Confusion matrix is printed to stdout during each training run and visible in Cloud Run logs. At F1=0.964 (macro), the dominant error mode is false negatives on the minority class — patients with CKD classified as healthy. This is the higher-risk error in a clinical screening context and should be monitored in production via the Firestore prediction logs.

---

## SHAP Feature Importance

SHAP (SHapley Additive exPlanations) values were computed using `shap.TreeExplainer` on the test set after training. The chart below shows the top 15 features ranked by mean absolute SHAP value.

![SHAP Feature Importance](shap_summary.png)

> `shap_summary.png` and `shap_values.json` are logged as MLflow artifacts on every successful Production promotion run.

**Key drivers of prediction (in approximate order of importance):**
- `hemo` — Haemoglobin level (strongest single predictor)
- `sc` — Serum creatinine (elevated in kidney failure)
- `sg` — Specific gravity of urine
- `al` — Albumin (protein in urine)
- `htn` — Hypertension co-morbidity
- `pcv` — Packed cell volume (anaemia indicator)
- `rc` — Red blood cell count
- `dm` — Diabetes mellitus co-morbidity

---

## Limitations

1. **Small dataset:** Only ~400 samples. Performance metrics may not generalise to larger or more diverse populations.
2. **High missingness:** Several categorical features (e.g., `rbc`, `pc`, `ba`) have significant missing rates in the source data. Imputation introduces uncertainty.
3. **No temporal validation:** All data is cross-sectional. The model has not been validated on prospective or longitudinal patient cohorts.
4. **External validity unknown:** The UCI dataset was collected from a single institution. Performance on other demographic groups or healthcare systems is untested.
5. **Binary output only:** The model does not output a calibrated probability — outputs should not be interpreted as clinical probabilities without further calibration.

---

## Ethical Considerations

- Predictions **must be reviewed by a qualified clinician** before informing any medical decision.
- The model is not approved for autonomous clinical use.
- Prediction logs are stored in Firestore (`prediction_logs` collection) for auditability.
- No personally identifiable information (PII) is transmitted to the API — input features are clinical measurements only.

---

## Infrastructure

| Component | Technology |
|---|---|
| Training | Python 3.10, scikit-learn, LightGBM, XGBoost |
| Experiment tracking | MLflow (Cloud Run) + Cloud SQL (PostgreSQL) |
| Artifact storage | Google Cloud Storage (`chronic-kd-mlflow-artifacts`) |
| Model serving | FastAPI + Uvicorn on Cloud Run |
| Prediction logging | Google Firestore |
| CI/CD | GitHub Actions (`train-and-register.yml`, `deploy-api.yml`) |
