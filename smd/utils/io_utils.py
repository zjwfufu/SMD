# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import boto3
import io
import os
import urllib
import hashlib
import requests
import html
import uuid
import glob
import re
from typing import Any

import torch

import smd.utils.logging_utils as logger


def check_grad_zero(model: torch.nn.Module) -> None:
    """Assert that all trainable parameters of ``model`` have zero gradient.

    Used by sanity harnesses to verify that a module which is supposed to be
    frozen (e.g. teacher network, or non-pivot student blocks under
    SimplifiedDistillation) really did not receive gradient flow.
    """
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.allclose(param.grad, torch.zeros_like(param.grad))


def set_env_vars(credentials_path: str = None) -> None:
    """
    Set the environment variables for SMD

    Args:
      credentials_path: The path to the JSON file containing AWS credentials and region information.
    """

    # Reads AWS credentials and configuration from a JSON file and sets them as environment variables.
    if credentials_path is not None and os.path.isfile(credentials_path):
        try:
            with open(credentials_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            key_map = {
                "AWS_ACCESS_KEY_ID": "aws_access_key_id",
                "AWS_SECRET_ACCESS_KEY": "aws_secret_access_key",
                "AWS_DEFAULT_REGION": "region_name",
                "AWS_ENDPOINT_URL": "endpoint_url",
                "S3_ENDPOINT_URL": "endpoint_url",
            }
            for env_key, config_key in key_map.items():
                if config_key in config:
                    os.environ[env_key] = config[config_key]
                else:
                    logger.warning(f"Missing key {config_key} in {credentials_path}, skipping env variable {env_key}.")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in {credentials_path}, skip loading AWS credentials.")
        else:
            logger.success(f"AWS credentials loaded from {credentials_path} and set as environment variables.")

    # HF cache resolution is handled by smd.utils.hf_utils.get_hf_cache_dir().
    # Runtime downloads use the official hub directory, while legacy HF_HOME and
    # project-local layouts remain read-compatible for existing caches.


def latest_checkpoint(path: str) -> str:
    """Return the latest checkpoint prefix with a committed ``.pth`` marker."""

    if path.startswith("s3://"):
        # Get the list of objects in the s3 container

        s3_client = boto3.client("s3")

        objects = s3_client.list_objects_v2(Bucket=path)["Contents"]
        checkpoint_entries = [obj["Key"] for obj in objects]
    elif os.path.exists(path):
        checkpoint_entries = os.listdir(path)
    else:
        checkpoint_entries = []

    committed_iterations = []
    artifact_iterations = []
    for entry in checkpoint_entries:
        filename = os.path.basename(entry)
        marker_match = re.fullmatch(r"(\d+)\.pth", filename)
        if marker_match is not None:
            committed_iterations.append(int(marker_match.group(1)))
            continue

        artifact_match = re.match(r"(\d+)\.", filename)
        if artifact_match is not None:
            artifact_iterations.append(int(artifact_match.group(1)))

    if not committed_iterations:
        logger.error(f"No model files found in {path}")
        return ""

    latest_iteration = max(committed_iterations)
    incomplete_iterations = sorted({iteration for iteration in artifact_iterations if iteration > latest_iteration})
    if incomplete_iterations:
        logger.warning(
            f"Ignoring incomplete checkpoint artifacts newer than committed iteration {latest_iteration}: "
            f"{incomplete_iterations}"
        )

    latest_model_path = os.path.join(path, f"{latest_iteration:07d}")

    return latest_model_path


def s3_load(s3_path: str) -> io.BytesIO:
    """Load a file from S3 bucket and return the content as bytes"""

    bucket = s3_path.split("/")[2]
    key = "/".join(s3_path.split("/")[3:])

    s3_client = boto3.client("s3")
    obj = s3_client.get_object(Bucket=bucket, Key=key)

    return io.BytesIO(obj["Body"].read())


def s3_save(s3_path: str, data: bytes) -> None:
    """Save a file to S3 bucket"""

    bucket = s3_path.split("/")[2]
    key = "/".join(s3_path.split("/")[3:])

    s3_client = boto3.client("s3")
    s3_client.put_object(Bucket=bucket, Key=key, Body=data)


def open_url(
    url: str,
    cache_dir: str = None,
    num_attempts: int = 10,
    verbose: bool = True,
    return_filename: bool = False,
    cache: bool = True,
) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match("^[a-z]+://", url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith("file://"):
        filename = urllib.parse.urlparse(url).path
        if re.match(r"^/[a-zA-Z]:", filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = os.path.join(".", ".cache")

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print(f"Downloading {url} ...", end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [
                                html.unescape(link) for link in content_str.split('"') if "export=download" in link
                            ]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        safe_name = safe_name[: min(len(safe_name), 128)]
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file)  # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)
