import math
from dataclasses import dataclass

import pandas as pd

from src.utils.exception import IngestError
from src.utils.main_utils import is_safe_identifier

NHANES_FILENAMES = (
    "BIOPRO_L.xpt",
    "BMX_L.xpt",
    "BPQ_L.xpt",
    "BPXO_L.xpt",
    "DEMO_L.xpt",
    "DIQ_L.xpt",
    "GHB_L.xpt",
    "HDL_L.xpt",
    "SMQ_L.xpt",
    "TCHOL_L.xpt",
)

TABLE_NAMES = {
    "BIOPRO_L.xpt": "BIOCHEMISTRY_PROFILES",
    "BMX_L.xpt": "BODY_MEASURES",
    "BPQ_L.xpt": "BLOOD_PRESSURE_CHOLESTEROL_QUESTIONNAIRES",
    "BPXO_L.xpt": "BLOOD_PRESSURE_READINGS",
    "DEMO_L.xpt": "DEMOGRAPHICS",
    "DIQ_L.xpt": "DIABETES_QUESTIONNAIRES",
    "GHB_L.xpt": "GLYCOHEMOGLOBIN_RESULTS",
    "HDL_L.xpt": "HDL_CHOLESTEROL_RESULTS",
    "SMQ_L.xpt": "SMOKING_QUESTIONNAIRES",
    "TCHOL_L.xpt": "TOTAL_CHOLESTEROL_RESULTS",
}

_SAS_TYPES = {"double": "numeric", "string": "char"}


@dataclass(frozen=True)
class Column:
    name: str
    sas_type: str
    label: str


@dataclass(frozen=True)
class ImportResult:
    filename: str
    table_name: str
    row_count: int


@dataclass
class Dataset:
    filename: str
    table_name: str
    columns: tuple[Column, ...]
    frame: pd.DataFrame

    def uses_seqn_primary_key(self) -> bool:
        if "seqn" not in self.frame.columns:
            return False
        seqn = self.frame["seqn"]
        return bool(seqn.notna().all() and seqn.nunique(dropna=True) == len(self.frame))

    def copy_records(self) -> list[tuple[float | str | None, ...]]:
        sas_types = {column.name: column.sas_type for column in self.columns}
        return [
            tuple(
                python_value(value, sas_types[str(column)])
                for column, value in zip(self.frame.columns, row, strict=True)
            )
            for row in self.frame.itertuples(index=False, name=None)
        ]


def table_name(filename: str) -> str:
    try:
        name = TABLE_NAMES[filename.rsplit("/", 1)[-1]]
    except KeyError as error:
        raise IngestError(f"No table name mapped for {filename}") from error
    if not is_safe_identifier(name):
        raise IngestError(f"Unsafe table name for {filename}")
    return name


def postgres_type(sas_type: str) -> str:
    if sas_type == "numeric":
        return "double precision"
    if sas_type == "char":
        return "text"
    raise IngestError(f"Unsupported SAS type: {sas_type}")


def storage_url(base_url: str, filename: str) -> str:
    return f"{base_url.rstrip('/')}/{filename}"


def columns_from_metadata(
    names: list[str],
    labels: list[str | None],
    variable_types: dict[str, str],
) -> tuple[Column, ...]:
    columns: list[Column] = []
    for name, label in zip(names, labels, strict=True):
        try:
            sas_type = _SAS_TYPES[variable_types[name]]
        except KeyError as error:
            raise IngestError(f"Unsupported SAS type for {name}") from error
        column_name = name.lower()
        if not is_safe_identifier(column_name):
            raise IngestError(f"Unsafe column name {name!r}")
        columns.append(Column(name=column_name, sas_type=sas_type, label=(label or "").strip()))
    return tuple(columns)


def python_value(value: object, sas_type: str) -> float | str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if sas_type == "char":
        if isinstance(value, bytes | bytearray):
            text = bytes(value).decode("latin1")
        else:
            text = str(value)
        return text.rstrip() or None
    number = float(value)  # type: ignore[arg-type]
    if math.isnan(number):
        return None
    return number
