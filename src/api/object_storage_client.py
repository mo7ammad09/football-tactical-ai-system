"""S3-compatible object storage helpers for videos and artifacts."""

from __future__ import annotations

import mimetypes
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Optional


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
        self.multipart_chunksize = 8 * 1024 * 1024
        self.max_upload_workers = 4

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
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
    ) -> Dict[str, Optional[str]]:
        """Upload a file using boto3 managed multipart transfer."""
        path = Path(local_path)
        if not path.exists() or path.stat().st_size <= 0:
            raise ValueError(f"File is missing or empty: {local_path}")

        key = object_key or self.object_key(file_kind=file_kind, file_name=path.name, job_id=job_id)
        guessed_type = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if progress_callback:
            self._upload_file_with_progress(path, key, guessed_type, progress_callback)
        else:
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

    def _emit_progress(
        self,
        progress_callback: Callable[[float, int, int], None],
        percent: float,
        transferred_bytes: int,
        total_bytes: int,
    ) -> None:
        """Call progress callbacks while keeping one-argument callbacks compatible."""
        try:
            progress_callback(percent, transferred_bytes, total_bytes)
        except TypeError:
            progress_callback(percent)  # type: ignore[misc]

    def _upload_file_with_progress(
        self,
        path: Path,
        key: str,
        content_type: str,
        progress_callback: Callable[[float, int, int], None],
    ) -> None:
        """Multipart upload with main-thread progress updates."""
        total_bytes = path.stat().st_size
        chunk_size = self.multipart_chunksize

        if total_bytes <= 5 * 1024 * 1024:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=path.read_bytes(),
                ContentType=content_type,
            )
            self._emit_progress(progress_callback, 100.0, total_bytes, total_bytes)
            return

        response = self.client.create_multipart_upload(
            Bucket=self.bucket,
            Key=key,
            ContentType=content_type,
        )
        upload_id = response["UploadId"]
        parts = []
        transferred = 0

        def upload_part(part_number: int, offset: int, size: int) -> Dict[str, Any]:
            with path.open("rb") as file_obj:
                file_obj.seek(offset)
                body = file_obj.read(size)
            part_response = self.client.upload_part(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=part_number,
                Body=body,
            )
            return {
                "PartNumber": part_number,
                "ETag": part_response["ETag"],
                "Size": len(body),
            }

        try:
            part_count = (total_bytes + chunk_size - 1) // chunk_size
            max_workers = max(1, min(self.max_upload_workers, part_count))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for part_number in range(1, part_count + 1):
                    offset = (part_number - 1) * chunk_size
                    size = min(chunk_size, total_bytes - offset)
                    futures.append(executor.submit(upload_part, part_number, offset, size))

                for future in as_completed(futures):
                    part = future.result()
                    parts.append({"PartNumber": part["PartNumber"], "ETag": part["ETag"]})
                    transferred += int(part["Size"])
                    percent = min(100.0, (transferred / total_bytes) * 100)
                    self._emit_progress(progress_callback, percent, transferred, total_bytes)

            parts.sort(key=lambda item: item["PartNumber"])
            self.client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
            self._emit_progress(progress_callback, 100.0, total_bytes, total_bytes)
        except Exception:
            self.client.abort_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
            )
            raise

    def download_file(self, object_key: str, local_path: str) -> None:
        """Download an object to a local path."""
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, object_key, str(dest))
