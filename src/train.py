import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
import matplotlib.pyplot as plt
import json
import numpy as np

from features import clean_raw_data, build_preprocessor

MODEL_CONFIGS = [
    {
        "name": "XGBoost",
        "clf": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    },
    {
        "name": "RandomForest",
        "clf": RandomForestClassifier(n_estimators=100, random_state=42)
    },
    {
        "name": "LightGBM",
        "clf": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    },
]


def log_shap_artifacts(pipeline, X_test, run_id):
    """Compute SHAP values for the winning pipeline and log chart + JSON to MLflow."""
    preprocessor = pipeline.named_steps['preprocessor']
    classifier   = pipeline.named_steps['classifier']

    X_test_transformed = preprocessor.transform(X_test)

    # Numeric feature names come directly from the transformer; categorical names are OHE-expanded
    numeric_features      = preprocessor.transformers_[0][2]
    categorical_transformer = preprocessor.transformers_[1][1]
    categorical_features    = preprocessor.transformers_[1][2]
    ohe_feature_names = categorical_transformer.get_feature_names_out(categorical_features).tolist()
    feature_names = list(numeric_features) + ohe_feature_names

    explainer   = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_transformed)

    # LightGBM binary classification may return a list of arrays (one per class); take class-1
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs   = np.abs(sv).mean(axis=0)
    importance = dict(zip(feature_names, mean_abs.tolist()))

    # --- Bar chart (top 15 features) ---
    sorted_idx = np.argsort(mean_abs)[-15:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i] for i in sorted_idx], mean_abs[sorted_idx], color='steelblue')
    ax.set_title("SHAP Feature Importance (Top 15)")
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    fig.savefig("shap_summary.png", dpi=150)
    plt.close(fig)

    with open("shap_values.json", "w") as f:
        json.dump(importance, f, indent=2)

    # Re-open the finished run just to log artifacts
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact("shap_summary.png")
        mlflow.log_artifact("shap_values.json")

    print("SHAP artifacts logged to MLflow.")
    return importance


def main():
    # Phase A: Data
    print("Downloading dataset from Google Cloud Storage...")
    train_df = pd.read_csv('gs://chronic-kd-kidney-disease-ml/raw_data/kidney_disease_train.csv')
    train_df = clean_raw_data(train_df)

    X = train_df.drop(columns=['classification'])
    y = train_df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Phase B: MLflow experiment setup + Production gate
    experiment_name = "ckd-cloud-experiment"
    try:
        mlflow.create_experiment(experiment_name, artifact_location="gs://chronic-kd-mlflow-artifacts/")
    except Exception:
        pass
    mlflow.set_experiment(experiment_name)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    best_past_f1 = 0.0
    if experiment is not None:
        past_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.stage = 'Production'",
            order_by=["metrics.f1_score DESC"],
            max_results=1
        )
        if past_runs:
            best_past_f1 = past_runs[0].data.metrics.get("f1_score", 0.0)

    print(f"Current Production F1 to beat: {best_past_f1:.3f}")

    # Phase C: Train all 3 models, one MLflow run each
    run_results = []

    for config in MODEL_CONFIGS:
        preprocessor = build_preprocessor()  # fresh instance — ColumnTransformer is stateful
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', config["clf"])
        ])

        with mlflow.start_run(run_name=config["name"]) as run:
            print(f"\nTraining {config['name']}...")
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            f1        = f1_score(y_test, y_pred, average='macro')
            precision = precision_score(y_test, y_pred, average='macro')
            recall    = recall_score(y_test, y_pred, average='macro')
            accuracy  = accuracy_score(y_test, y_pred)
            cm        = confusion_matrix(y_test, y_pred)

            print(f"  {config['name']} -> F1: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")
            print(f"  Confusion Matrix:\n{cm}")

            mlflow.log_param("model_type", config["name"])
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(pipeline, "model")
            mlflow.set_tag("stage", "challenger")

            run_results.append({
                "run_id": run.info.run_id,
                "name": config["name"],
                "f1": f1,
                "pipeline": pipeline
            })

    # Phase D: Pick winner and apply Production gate
    best_new = max(run_results, key=lambda r: r["f1"])
    print(f"\nBest new model: {best_new['name']} (F1={best_new['f1']:.3f})")

    if best_new["f1"] > best_past_f1:
        print(f"Evaluation Gate PASSED — promoting {best_new['name']} to Production.")
        client.set_tag(best_new["run_id"], "stage", "Production")
        for r in run_results:
            if r["run_id"] != best_new["run_id"]:
                client.set_tag(r["run_id"], "stage", "rejected")

        # Register to MLflow Model Registry and set Production alias
        MODEL_REGISTRY_NAME = "kidney-disease-baseline"
        registered = mlflow.register_model(
            model_uri=f"runs:/{best_new['run_id']}/model",
            name=MODEL_REGISTRY_NAME
        )
        client.set_registered_model_alias(MODEL_REGISTRY_NAME, "Production", registered.version)
        print(f"Registered version {registered.version} with alias 'Production'")

        # Compute and log SHAP feature importance for the winning model
        print("\nComputing SHAP values...")
        log_shap_artifacts(best_new["pipeline"], X_test, best_new["run_id"])

    else:
        print(f"Evaluation Gate FAILED — no new model beats Production F1 of {best_past_f1:.3f}.")
        for r in run_results:
            client.set_tag(r["run_id"], "stage", "rejected")

if __name__ == "__main__":
    main()
