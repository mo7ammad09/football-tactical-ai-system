"""S3-compatible object storage helpers for videos and artifacts."""

from __future__ import annotations

import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


class ObjectStorageClient:
    """Small wrapper around boto3 for S3-compatible storage."""

    def __init__(
        self,
        bucket: Optional[str] = None,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        prefix: Optional[str] = None,
        public_base_url: Optional[str] = None,
    ):
        self.bucket = (bucket or os.environ.get("OBJECT_STORAGE_BUCKET", "")).strip()
        self.region = (region or os.environ.get("OBJECT_STORAGE_REGION", "us-east-1")).strip() or "us-east-1"
        self.endpoint_url = (endpoint_url or os.environ.get("OBJECT_STORAGE_ENDPOINT_URL", "")).strip() or None
        self.prefix = (prefix or os.environ.get("OBJECT_STORAGE_PREFIX", "football-ai")).strip().strip("/")
        self.public_base_url = (
            public_base_url or os.environ.get("OBJECT_STORAGE_PUBLIC_BASE_URL", "")
        ).strip().rstrip("/")
        self.presign_expires = int(
            os.environ.get("OBJECT_STORAGE_PRESIGN_EXPIRES")
            or os.environ.get("OBJECT_STORAGE_URL_EXPIRES", "86400")
        )

        if not self.bucket:
            raise ValueError("OBJECT_STORAGE_BUCKET is required for object storage")

        try:
            import boto3
            from boto3.s3.transfer import TransferConfig
            from botocore.config import Config as BotoConfig
        except ImportError as exc:
            raise RuntimeError("boto3 is required for object storage. Install boto3 or server requirements.") from exc

        access_key = os.environ.get("OBJECT_STORAGE_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("OBJECT_STORAGE_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
        session_token = os.environ.get("OBJECT_STORAGE_SESSION_TOKEN") or os.environ.get("AWS_SESSION_TOKEN")

        kwargs: Dict[str, Any] = {
            "service_name": "s3",
            "region_name": self.region,
            "endpoint_url": self.endpoint_url,
            "config": BotoConfig(signature_version="s3v4"),
        }
        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key
        if session_token:
            kwargs["aws_session_token"] = session_token

        self.client = boto3.client(**kwargs)
        self.transfer_config = TransferConfig(
            multipart_threshold=32 * 1024 * 1024,
            multipart_chunksize=16 * 1024 * 1024,
            max_concurrency=4,
            use_threads=True,
        )

    def object_key(self, file_kind: str, file_name: str, job_id: Optional[str] = None) -> str:
        """Build a canonical object key."""
        safe_name = Path(file_name or "upload.bin").name or "upload.bin"
        prefix = self.prefix.strip("/")
        if job_id:
            key = f"artifacts/{job_id}/{safe_name}"
        else:
            key = f"{file_kind}/{uuid.uuid4()}_{safe_name}"
        return f"{prefix}/{key}" if prefix else key

    def public_url(self, object_key: str) -> Optional[str]:
        """Return a public or presigned URL for a stored object."""
        if self.public_base_url:
            return f"{self.public_base_url}/{object_key}"
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": object_key},
                ExpiresIn=self.presign_expires,
            )
        except Exception:
            return None

    def upload_file(
        self,
        local_path: str,
        *,
        file_kind: str = "video",
        object_key: Optional[str] = None,
        content_type: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        """Upload a file using boto3 managed multipart transfer."""
        path = Path(local_path)
        if not path.exists() or path.stat().st_size <= 0:
            raise ValueError(f"File is missing or empty: {local_path}")

        key = object_key or self.object_key(file_kind=file_kind, file_name=path.name, job_id=job_id)
        guessed_type = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.client.upload_file(
            str(path),
            self.bucket,
            key,
            ExtraArgs={"ContentType": guessed_type},
            Config=self.transfer_config,
        )
        return {
            "object_key": key,
            "object_uri": f"s3://{self.bucket}/{key}",
            "public_url": self.public_url(key),
        }

    def download_file(self, object_key: str, local_path: str) -> None:
        """Download an object to a local path."""
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, object_key, str(dest))
