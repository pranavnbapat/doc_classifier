from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List

from openpyxl import load_workbook

from docint.ingest.models import IngestedAsset


TABULAR_MAX_ROWS = int(os.getenv("TABULAR_MAX_ROWS", "100"))
TABULAR_PREVIEW_ROWS = int(os.getenv("TABULAR_PREVIEW_ROWS", "30"))
XLSX_MAX_SHEETS = int(os.getenv("XLSX_MAX_SHEETS", "10"))
XLSX_MAX_ROWS_PER_SHEET = int(os.getenv("XLSX_MAX_ROWS_PER_SHEET", "25"))


def _stringify_row(values: List[object]) -> str:
    cleaned = [str(v).strip() for v in values if v is not None and str(v).strip()]
    return " | ".join(cleaned)


def ingest_delimited_file(file_path: str, filename: str, delimiter: str) -> IngestedAsset:
    path = Path(file_path)
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for idx, row in enumerate(reader):
            rows.append(row)
            if idx >= max(1, TABULAR_MAX_ROWS) - 1:
                break

    lines: List[str] = []
    if rows:
        header = _stringify_row(rows[0])
        if header:
            lines.append(f"Columns: {header}")
        for idx, row in enumerate(rows[1: max(1, TABULAR_PREVIEW_ROWS) + 1], start=1):
            rendered = _stringify_row(row)
            if rendered:
                lines.append(f"Row {idx}: {rendered}")

    text = "\n".join(lines).strip()
    suffix = path.suffix.lower()
    mime = "text/csv" if suffix == ".csv" else "text/tab-separated-values"

    return IngestedAsset(
        asset_type=suffix.lstrip("."),
        filename=filename,
        source_path=file_path,
        text=text,
        lines=lines,
        units=max(1, len(rows) - 1 if rows else 1),
        unit_label="rows",
        source="tabular_text",
        mime_type=mime,
        visual_candidate=False,
        ocr_supported=False,
        meta={
            "delimiter": delimiter,
            "preview_rows": max(0, len(rows) - 1),
            "column_count": len(rows[0]) if rows else 0,
        },
    )


def ingest_xlsx(file_path: str, filename: str) -> IngestedAsset:
    workbook = load_workbook(file_path, read_only=True, data_only=True)
    lines: List[str] = []
    total_rows = 0
    max_columns = 0
    sheet_names = workbook.sheetnames

    for sheet_name in sheet_names[: max(1, XLSX_MAX_SHEETS)]:
        ws = workbook[sheet_name]
        lines.append(f"Sheet: {sheet_name}")
        preview_count = 0
        for row in ws.iter_rows(values_only=True):
            max_columns = max(max_columns, len([cell for cell in row if cell is not None and str(cell).strip()]))
            rendered = _stringify_row(list(row))
            if rendered:
                label = "Columns" if preview_count == 0 else f"Row {preview_count}"
                lines.append(f"{label}: {rendered}")
                preview_count += 1
                total_rows += 1
            if preview_count >= max(1, XLSX_MAX_ROWS_PER_SHEET):
                break

    text = "\n".join(lines).strip()

    return IngestedAsset(
        asset_type="xlsx",
        filename=filename,
        source_path=file_path,
        text=text,
        lines=lines,
        units=max(1, total_rows),
        unit_label="rows",
        source="xlsx_text",
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        visual_candidate=False,
        ocr_supported=False,
        meta={
            "sheet_count": len(sheet_names),
            "sheet_names": sheet_names,
            "preview_rows": total_rows,
            "max_columns": max_columns,
        },
    )
