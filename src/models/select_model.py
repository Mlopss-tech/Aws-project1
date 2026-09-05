import json
import os
import mlflow
from mlflow.tracking import MlflowClient


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "telco-churn-model"
MIN_ROC_AUC = 0.80

OUTPUT_FILE = "selected_model.json"


# ============================================================
# SELECT BEST MODEL
# ============================================================

def select_best_model():

    print(f"🔎 Searching MLflow model: {MODEL_NAME}")
    print(f"🎯 Minimum ROC-AUC: {MIN_ROC_AUC}")

    client = MlflowClient()

    # Get all registered model versions
    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not versions:
        raise Exception(
            f"❌ No versions found for model: {MODEL_NAME}"
        )

    print(f"\n📦 Found {len(versions)} model versions")

    qualified_models = []

    # --------------------------------------------------------
    # Check every model version
    # --------------------------------------------------------

    for version in versions:

        version_number = int(version.version)
        run_id = version.run_id

        print(f"\nChecking model version {version_number}...")

        try:
            # Get ROC-AUC from MLflow run
            run = client.get_run(run_id)

            metrics = run.data.metrics

            roc_auc = metrics.get("roc_auc")

            # Some projects may use roc_auc_score
            if roc_auc is None:
                roc_auc = metrics.get("roc_auc_score")

            if roc_auc is None:
                print("   ⚠️ ROC-AUC metric not found")
                continue

            print(f"   ROC-AUC: {roc_auc:.3f}")

            # ------------------------------------------------
            # Quality Gate
            # ------------------------------------------------

            if roc_auc >= MIN_ROC_AUC:

                print("   ✅ Passed quality threshold")

                qualified_models.append(
                    {
                        "version": version_number,
                        "run_id": run_id,
                        "roc_auc": float(roc_auc),
                        "model_name": MODEL_NAME
                    }
                )

            else:

                print("   ❌ Failed quality threshold")

        except Exception as e:

            print(
                f"   ⚠️ Could not evaluate version "
                f"{version_number}: {e}"
            )

    # ========================================================
    # Check if any model passed
    # ========================================================

    if not qualified_models:

        raise Exception(
            f"\n❌ No model passed ROC-AUC threshold "
            f"{MIN_ROC_AUC}"
        )

    # ========================================================
    # SELECT BEST MODEL
    #
    # 1. Highest ROC-AUC
    # 2. If ROC-AUC is tied → newest version
    # ========================================================

    best_model = max(
        qualified_models,
        key=lambda x: (
            x["roc_auc"],
            x["version"]
        )
    )

    # ========================================================
    # S3 LOCATION
    # ========================================================

    bucket_name = "telco-mlops-yourname-2026"

    s3_path = (
        f"s3://{bucket_name}/"
        f"models/{MODEL_NAME}/"
        f"run-{best_model['run_id']}/"
    )

    best_model["s3_path"] = s3_path

    # ========================================================
    # PRINT RESULT
    # ========================================================

    print("\n" + "=" * 60)
    print("🏆 BEST MODEL SELECTED")
    print("=" * 60)

    print(f"Model       : {best_model['model_name']}")
    print(f"Version     : {best_model['version']}")
    print(f"Run ID      : {best_model['run_id']}")
    print(f"ROC-AUC     : {best_model['roc_auc']:.3f}")
    print(f"Threshold   : {MIN_ROC_AUC:.3f}")
    print(f"S3 Path     : {best_model['s3_path']}")

    print("=" * 60)

    # ========================================================
    # SAVE SELECTION RESULT
    # ========================================================

    with open(OUTPUT_FILE, "w") as f:
        json.dump(best_model, f, indent=4)

    print(f"\n💾 Selected model information saved to:")
    print(f"   {os.path.abspath(OUTPUT_FILE)}")

    print("\nSelected model information:")
    print(best_model)

    return best_model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:
        select_best_model()

    except Exception as e:

        print(f"\n❌ Model selection failed:")
        print(e)

        raise