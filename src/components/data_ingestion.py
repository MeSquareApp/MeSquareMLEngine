"""Download NHANES transport files and turn them into datasets."""

import tempfile
import urllib.request
from pathlib import Path

import pandas as pd
import pyreadstat

from src.components.data_transformation import (
    Column,
    Dataset,
    columns_from_metadata,
    storage_url,
    table_name,
)


def download(url: str) -> bytes:
    """Fetch the bytes at a public NHANES URL."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "MeSquareMLEngine/0.1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return bytes(response.read())


def read_xpt(path: Path) -> tuple[pd.DataFrame, tuple[Column, ...]]:
    """Read a SAS transport file into a frame and column metadata."""
    frame, metadata = pyreadstat.read_xport(path, encoding="latin1")
    columns = columns_from_metadata(
        list(metadata.column_names),
        list(metadata.column_labels),
        dict(metadata.readstat_variable_types),
    )
    frame.columns = [column.name for column in columns]
    return frame, columns


def read_xpt_bytes(payload: bytes) -> tuple[pd.DataFrame, tuple[Column, ...]]:
    """Read SAS transport bytes without leaving a file on disk."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "dataset.xpt"
        path.write_bytes(payload)
        return read_xpt(path)


def fetch_dataset(base_url: str, filename: str) -> Dataset:
    """Download one NHANES file and name its Postgres table."""
    frame, columns = read_xpt_bytes(download(storage_url(base_url, filename)))
    return Dataset(
        filename=filename,
        table_name=table_name(filename),
        columns=columns,
        frame=frame,
    )
