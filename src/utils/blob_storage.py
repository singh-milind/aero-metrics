from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import io
import joblib
import pandas as pd
import json

STORAGE_ACCOUNT_NAME = "aerometricsstrorage"

account_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"

credential = DefaultAzureCredential()

blob_service_client = BlobServiceClient(
    account_url=account_url,
    credential=credential,
)


def upload_csv(
    container_name: str,
    blob_name: str,
    csv_data: str
):
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name,
    )

    blob_client.upload_blob(
        csv_data.encode("utf-8"),
        overwrite=True
    )



def load_csv(
    container_name: str,
    blob_name: str
) -> pd.DataFrame:

    csv_bytes = download_blob_bytes(
        container_name=container_name,
        blob_name=blob_name
    )

    df = pd.read_csv(
        io.BytesIO(csv_bytes)
    )

    return df



def download_blob_bytes(container_name: str, blob_name: str) -> bytes:
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name,
    )

    return blob_client.download_blob().readall()



def upload_model(
    container_name: str,
    blob_name: str,
    model,
    logger,
    artifact_name: str
):
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name,
    )

    # Serialize model into memory
    buffer = io.BytesIO()

    joblib.dump(model, buffer)

    # Move to the beginning of the buffer
    buffer.seek(0)

    # Upload directly to Azure Blob Storage
    blob_client.upload_blob(
        buffer,
        overwrite=True
    )

    logger.info(
        f"Model uploaded to Azure: {artifact_name} - "
        f"{container_name}/{blob_name}"
    )
    
def upload_metric(
    container_name: str,
    blob_name: str,
    metric: dict,
    logger,
    artifact_name: str,
):
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name,
    )
    
    # Convert existing dictionary into an in-memory buffer
    
    json_data = json.dumps(metric,indent=4)
    
    buffer = io.BytesIO(
        json_data.encode("utf-8")
    )

    # Move to the beginning of the buffer
    buffer.seek(0)

    # Upload JSON directly to Azure Blob Storage
    blob_client.upload_blob(
        buffer,
        overwrite=True,
    )

    logger.info(
        f"Metric uploaded to Azure: {artifact_name} - "
        f"{container_name}/{blob_name}"
    )