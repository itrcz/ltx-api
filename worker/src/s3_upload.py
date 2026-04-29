"""S3-compatible upload. Works with AWS S3, Cloudflare R2, Backblaze B2, etc."""
import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config


def _client():
    endpoint = os.environ.get("S3_ENDPOINT_URL") or None
    region = os.environ.get("S3_REGION", "auto")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )


def upload_and_presign(
    local_path: Path,
    key: str,
    expires_sec: int = 3600,
    content_type: str = "video/mp4",
) -> str:
    bucket = os.environ["S3_BUCKET"]
    s3 = _client()
    s3.upload_file(
        str(local_path), bucket, key,
        ExtraArgs={"ContentType": content_type},
    )
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_sec,
    )


def upload_bytes(
    data: bytes,
    key: str,
    content_type: str = "application/octet-stream",
) -> None:
    """Upload an in-memory bytes blob to s3://{bucket}/{key}. No presign —
    used for sidecar files (e.g. result.json) where the canonical access
    path is by key, not by signed URL."""
    bucket = os.environ["S3_BUCKET"]
    s3 = _client()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
