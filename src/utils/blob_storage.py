from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

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

def upload_directory(container_name: str, local_dir: str, blob_prefix: str = ""):
    local_dir = Path(local_dir)

    for file_path in local_dir.rglob("*"):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(local_dir).as_posix()

        if blob_prefix:
            blob_name = f"{blob_prefix}/{relative_path}"
        else:
            blob_name = relative_path

        upload_blob(
            container_name,
            blob_name,
            str(file_path),
        )