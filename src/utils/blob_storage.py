from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import io
import joblib
import pandas as pd

STORAGE_ACCOUNT_NAME = "aerometricsstrorage"

account_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"

credential = DefaultAzureCredential()

blob_service_client = BlobServiceClient(
    account_url=account_url,
    credential=credential,
)



def download_blob(container_name: str, blob_name: str, local_path: str):
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name,
    )

    with open(local_path, "wb") as file:
        file.write(blob_client.download_blob().readall())

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


def upload_blob(container_name: str, blob_name: str, local_path: str):
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name,
    )

    with open(local_path, "rb") as file:
        blob_client.upload_blob(file, overwrite=True)

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
