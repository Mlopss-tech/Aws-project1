#!/usr/bin/env python3

"""
Runs sequentially:
load → validate → preprocess → feature engineering
→ train/test split → XGBoost training → evaluation → MLflow
"""

import os
import sys
import time
import argparse

import pandas as pd
import mlflow
import mlflow.sklearn

# NEW: AWS S3 support
import boto3
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ============================================================
# Fix import path for local modules
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


# ============================================================
# Local modules - Core pipeline components
# ============================================================

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data


# ============================================================
# Main pipeline
# ============================================================

def main(args):

    # ========================================================
    # MLflow Setup
    # ========================================================

    from pathlib import Path

    project_root = Path(
        __file__
    ).resolve().parent.parent

    mlruns_path = project_root / "mlruns"

    mlflow.set_tracking_uri(
        mlruns_path.as_uri()
    )

    mlflow.set_experiment(
        args.experiment
    )


    # ========================================================
    # Start MLflow run
    # ========================================================

    with mlflow.start_run():

        # ====================================================
        # Log hyperparameters
        # ====================================================

        mlflow.log_param(
            "model",
            "xgboost"
        )

        mlflow.log_param(
            "threshold",
            args.threshold
        )

        mlflow.log_param(
            "test_size",
            args.test_size
        )


        # ====================================================
        # STAGE 1: Data Loading
        # ====================================================

        print(
            "🔄 Loading data..."
        )

        df = load_data(
            args.input
        )

        print(
            f"✅ Data loaded: "
            f"{df.shape[0]} rows, "
            f"{df.shape[1]} columns"
        )


        # ====================================================
        # Data Validation
        # ====================================================

        print(
            "🔍 Validating data quality "
            "with Great Expectations..."
        )

        is_valid, failed = (
            validate_telco_data(df)
        )

        mlflow.log_metric(
            "data_quality_pass",
            int(is_valid)
        )

        if not is_valid:

            import json

            mlflow.log_text(
                json.dumps(
                    failed,
                    indent=2
                ),
                artifact_file=(
                    "failed_expectations.json"
                )
            )

            raise ValueError(
                f"❌ Data quality check failed. "
                f"Issues: {failed}"
            )

        else:

            print(
                "✅ Data validation passed. "
                "Logged to MLflow."
            )


        # ====================================================
        # STAGE 2: Data Preprocessing
        # ====================================================

        print(
            "🔧 Preprocessing data..."
        )

        df = preprocess_data(
            df
        )


        # ====================================================
        # Save processed dataset LOCALLY
        # ====================================================

        processed_path = os.path.join(
            project_root,
            "data",
            "processed",
            "telco_churn_processed.csv"
        )

        os.makedirs(
            os.path.dirname(
                processed_path
            ),
            exist_ok=True
        )

        df.to_csv(
            processed_path,
            index=False
        )

        print(
            f"✅ Processed dataset saved "
            f"locally to {processed_path} "
            f"| Shape: {df.shape}"
        )


        # ====================================================
        # NEW: Upload processed dataset to AWS S3
        # ====================================================

        print(
            "☁️ Uploading processed dataset to S3..."
        )

        # Your AWS S3 bucket
        S3_BUCKET = (
            "telco-mlops-yourname-2026"
        )

        # Location inside S3
        S3_KEY = (
            "processed/"
            "telco_churn_processed.csv"
        )

        # Create S3 client
        s3 = boto3.client(
            "s3"
        )

        # Convert DataFrame to CSV in memory
        csv_buffer = StringIO()

        df.to_csv(
            csv_buffer,
            index=False
        )

        # Upload CSV to S3
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=S3_KEY,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv"
        )

        processed_s3_path = (
            f"s3://{S3_BUCKET}/{S3_KEY}"
        )

        print(
            "☁️ Processed dataset uploaded "
            "successfully:"
        )

        print(
            f"   {processed_s3_path}"
        )

        # Log S3 location in MLflow
        mlflow.log_param(
            "processed_data_s3",
            processed_s3_path
        )


        # ====================================================
        # STAGE 3: Feature Engineering
        # ====================================================

        print(
            "🛠️ Building features..."
        )

        target = args.target

        if target not in df.columns:

            raise ValueError(
                f"Target column "
                f"'{target}' not found in data"
            )

        df_enc = build_features(
            df,
            target_col=target
        )


        # ====================================================
        # Convert boolean columns
        # ====================================================

        for c in (
            df_enc
            .select_dtypes(
                include=["bool"]
            )
            .columns
        ):

            df_enc[c] = (
                df_enc[c]
                .astype(int)
            )

        print(
            f"✅ Feature engineering "
            f"completed: "
            f"{df_enc.shape[1]} features"
        )


        # ====================================================
        # Save Feature Metadata
        # ====================================================

        import json
        import joblib

        artifacts_dir = os.path.join(
            project_root,
            "artifacts"
        )

        os.makedirs(
            artifacts_dir,
            exist_ok=True
        )


        # Get feature columns
        feature_cols = list(
            df_enc
            .drop(
                columns=[target]
            )
            .columns
        )


        # Save feature columns locally
        with open(
            os.path.join(
                artifacts_dir,
                "feature_columns.json"
            ),
            "w"
        ) as f:

            json.dump(
                feature_cols,
                f
            )


        # Log feature columns to MLflow
        mlflow.log_text(
            "\n".join(
                feature_cols
            ),
            artifact_file=(
                "feature_columns.txt"
            )
        )


        # ====================================================
        # Save preprocessing artifact
        # ====================================================

        preprocessing_artifact = {

            "feature_columns":
                feature_cols,

            "target":
                target
        }

        preprocessing_path = os.path.join(
            artifacts_dir,
            "preprocessing.pkl"
        )

        joblib.dump(
            preprocessing_artifact,
            preprocessing_path
        )

        mlflow.log_artifact(
            preprocessing_path
        )

        print(
            f"✅ Saved "
            f"{len(feature_cols)} "
            f"feature columns "
            f"for serving consistency"
        )


        # ====================================================
        # STAGE 4: Train/Test Split
        # ====================================================

        print(
            "📊 Splitting data..."
        )

        X = df_enc.drop(
            columns=[target]
        )

        y = df_enc[target]


        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=args.test_size,
                stratify=y,
                random_state=42
            )
        )

        print(
            f"✅ Train: "
            f"{X_train.shape[0]} samples "
            f"| Test: "
            f"{X_test.shape[0]} samples"
        )


        # ====================================================
        # Class Imbalance
        # ====================================================

        scale_pos_weight = (
            (y_train == 0).sum()
            /
            (y_train == 1).sum()
        )

        print(
            f"📈 Class imbalance ratio: "
            f"{scale_pos_weight:.2f} "
            f"(applied to positive class)"
        )


        # ====================================================
        # STAGE 5: XGBoost Training
        # ====================================================

        print(
            "🤖 Training XGBoost model..."
        )

        model = XGBClassifier(

            n_estimators=301,

            learning_rate=0.034,

            max_depth=7,

            subsample=0.95,

            colsample_bytree=0.98,

            n_jobs=-1,

            random_state=42,

            eval_metric="logloss",

            scale_pos_weight=(
                scale_pos_weight
            )
        )


        # ====================================================
        # Train model
        # ====================================================

        t0 = time.time()

        model.fit(
            X_train,
            y_train
        )

        train_time = (
            time.time() - t0
        )

        mlflow.log_metric(
            "train_time",
            train_time
        )

        print(
            f"✅ Model trained in "
            f"{train_time:.2f} seconds"
        )


        # ====================================================
        # STAGE 6: Model Evaluation
        # ====================================================

        print(
            "📊 Evaluating model performance..."
        )

        t1 = time.time()

        proba = (
            model
            .predict_proba(
                X_test
            )[:, 1]
        )

        y_pred = (
            proba >= args.threshold
        ).astype(int)

        pred_time = (
            time.time() - t1
        )

        mlflow.log_metric(
            "pred_time",
            pred_time
        )


        # ====================================================
        # Metrics
        # ====================================================

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            proba
        )


        # ====================================================
        # Log metrics to MLflow
        # ====================================================

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )


        # ============================================================
        # MODEL QUALITY GATE
        # ============================================================

        QUALITY_THRESHOLD = 0.80

        print(
            f"\n🔎 Model Quality Gate:"
        )
        print(
            f"   ROC-AUC = {roc_auc:.3f}"
        )
        print(
            f"   Required ROC-AUC >= {QUALITY_THRESHOLD:.2f}"
        )

        if roc_auc < QUALITY_THRESHOLD:

            print(
                "❌ MODEL QUALITY GATE FAILED"
            )

            print(
                f"ROC-AUC {roc_auc:.3f} is below "
                f"the required threshold "
                f"{QUALITY_THRESHOLD:.2f}"
            )

            # Log gate result to MLflow
            mlflow.log_metric(
                "quality_gate_passed",
                0
            )

            # Stop pipeline
            raise RuntimeError(
                "Model rejected because it "
                "did not pass the quality gate."
            )

        else:

            print(
                "✅ MODEL QUALITY GATE PASSED"
            )

            print(
                f"ROC-AUC {roc_auc:.3f} >= "
                f"{QUALITY_THRESHOLD:.2f}"
            )

            mlflow.log_metric(
                "quality_gate_passed",
                1
            )


        # ====================================================
        # Print Model Performance
        # ====================================================

        print(
            "🎯 Model Performance:"
        )

        print(
            f"   Precision: "
            f"{precision:.3f} "
            f"| Recall: "
            f"{recall:.3f}"
        )

        print(
            f"   F1 Score: "
            f"{f1:.3f} "
            f"| ROC AUC: "
            f"{roc_auc:.3f}"
        )

        # ============================================================
        # STAGE 7: Save + Register Model in MLflow
        # ============================================================

        print("💾 Saving model to MLflow...")

        MODEL_NAME = "telco-churn-model"

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        print(
            f"✅ Model saved and registered in MLflow: "
            f"{MODEL_NAME}"
        )


        # ============================================================
        # STAGE 8: Upload Model Artifact to AWS S3
        # ============================================================

        print("☁️ Uploading model artifact to AWS S3...")


        # AWS S3 configuration
        S3_BUCKET = "telco-mlops-yourname-2026"


        # Get the current MLflow run ID
        run_id = mlflow.active_run().info.run_id


        # S3 location for this model version
        S3_MODEL_PREFIX = (
            f"models/"
            f"{MODEL_NAME}/"
            f"run-{run_id}/"
        )


        # Create temporary local model directory
        model_artifact_dir = os.path.join(
            project_root,
            "artifacts",
            "model"
        )

        os.makedirs(
            model_artifact_dir,
            exist_ok=True
        )


        # Save the trained XGBoost model locally (native format)
        model_file = os.path.join(
            model_artifact_dir,
            "model.json"
        )

        model.save_model(model_file)


        # NEW: Save the trained model as a pickle (joblib) file too
        # This preserves the full XGBClassifier wrapper (predict_proba,
        # hyperparams, scale_pos_weight, etc.) for fast Python-native reload.
        model_pkl_file = os.path.join(
            model_artifact_dir,
            "model.pkl"
        )

        joblib.dump(
            model,
            model_pkl_file
        )

        print(
            f"✅ Model also saved as pickle: "
            f"{model_pkl_file}"
        )


        # Save feature columns needed for inference
        feature_columns_file = os.path.join(
            model_artifact_dir,
            "feature_columns.json"
        )

        with open(
            feature_columns_file,
            "w"
        ) as f:
            json.dump(
                feature_cols,
                f,
                indent=4
            )


        # Create S3 client
        s3 = boto3.client("s3")


        # Upload model (native JSON format)
        s3.upload_file(
            model_file,
            S3_BUCKET,
            S3_MODEL_PREFIX + "model.json"
        )


        # NEW: Upload model (pickle format)
        s3.upload_file(
            model_pkl_file,
            S3_BUCKET,
            S3_MODEL_PREFIX + "model.pkl"
        )


        # Upload feature metadata
        s3.upload_file(
            feature_columns_file,
            S3_BUCKET,
            S3_MODEL_PREFIX + "feature_columns.json"
        )


        # Create model metadata
        model_metadata = {
            "model_name": MODEL_NAME,
            "run_id": run_id,
            "roc_auc": float(roc_auc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "quality_threshold": QUALITY_THRESHOLD,
            "quality_gate": "PASSED",
            "s3_model_path": (
                f"s3://{S3_BUCKET}/"
                f"{S3_MODEL_PREFIX}"
            )
        }


        metadata_file = os.path.join(
            model_artifact_dir,
            "model_metadata.json"
        )

        with open(
            metadata_file,
            "w"
        ) as f:
            json.dump(
                model_metadata,
                f,
                indent=4
            )


        # Upload metadata
        s3.upload_file(
            metadata_file,
            S3_BUCKET,
            S3_MODEL_PREFIX + "model_metadata.json"
        )


        model_s3_path = (
            f"s3://{S3_BUCKET}/"
            f"{S3_MODEL_PREFIX}"
        )


        print(
            "✅ Model uploaded successfully to S3:"
        )

        print(
            f"   {model_s3_path}"
        )


        # Log S3 model location in MLflow
        mlflow.log_param(
            "model_s3_path",
            model_s3_path
        )
# ============================================================
# Command-line arguments
# ============================================================

if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description=(
            "Run churn pipeline "
            "with XGBoost + MLflow + S3"
        )
    )

    p.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "path to CSV "
            "(local path or s3:// path)"
        )
    )

    p.add_argument(
        "--target",
        type=str,
        default="Churn"
    )

    p.add_argument(
        "--threshold",
        type=float,
        default=0.35
    )

    p.add_argument(
        "--test_size",
        type=float,
        default=0.2
    )

    p.add_argument(
        "--experiment",
        type=str,
        default="Telco Churn"
    )

    p.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        help=(
            "override MLflow tracking URI, "
            "else uses project_root/mlruns"
        )
    )

    args = p.parse_args()

    main(args)