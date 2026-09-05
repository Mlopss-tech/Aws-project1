import pandas as pd
import boto3
from io import BytesIO


# Your actual S3 bucket name
BUCKET_NAME = "telco-mlops-yourname-2026"


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads CSV data from either a local path or an S3 path.

    Args:
        file_path: Local CSV path or S3 URI.

    Returns:
        pandas DataFrame.
    """

    # If the path is an S3 location
    if file_path.startswith("s3://"):

        # Remove s3://
        s3_path = file_path.replace("s3://", "", 1)

        # Split bucket and file path
        bucket, key = s3_path.split("/", 1)

        # Create S3 client
        s3 = boto3.client("s3")

        # Download object from S3
        response = s3.get_object(
            Bucket=bucket,
            Key=key
        )

        # Convert S3 response into pandas DataFrame
        df = pd.read_csv(
            BytesIO(response["Body"].read())
        )

        return df

    # Otherwise, load from local filesystem
    return pd.read_csv(file_path)


if __name__ == "__main__":

    # Your CSV is currently directly inside the bucket
    file_path = (
        f"s3://{BUCKET_NAME}/Telco-Customer-Churn.csv"
    )

    # Load data
    df = load_data(file_path)

    # Display results
    print("Data loaded successfully!")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())