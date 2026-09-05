# 


import pandas as pd
import boto3
from io import BytesIO


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads CSV data from either a local path or an S3 path.

    Args:
        file_path: Local file path or S3 URI.

    Returns:
        Loaded pandas DataFrame.
    """

    if file_path.startswith("s3://"):
        # Example:
        # s3://my-bucket/raw/Telco-Customer-Churn.csv

        s3_path = file_path.replace("s3://", "", 1)
        bucket, key = s3_path.split("/", 1)

        s3 = boto3.client("s3")

        response = s3.get_object(
            Bucket=bucket,
            Key=key
        )

        return pd.read_csv(BytesIO(response["Body"].read()))

    # Existing local-file behavior
    return pd.read_csv(file_path)