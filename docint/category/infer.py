from __future__ import annotations

from dataclasses import dataclass

from docint.ingest.models import IngestedAsset


@dataclass
class CategoryInferenceResult:
    category: str
    confidence: float
    rationale: str


def infer_category(asset: IngestedAsset) -> CategoryInferenceResult:
    if asset.asset_type in {"pdf", "txt", "docx", "pptx"}:
        return CategoryInferenceResult(
            category="Document",
            confidence=0.95,
            rationale=f"File type .{asset.asset_type} is currently routed as Document",
        )

    if asset.asset_type in {"jpeg", "jpg", "png"}:
        return CategoryInferenceResult(
            category="Image",
            confidence=0.98,
            rationale=f"Image file type .{asset.asset_type} is routed as Image",
        )

    if asset.asset_type in {"mp3", "wav", "m4a"}:
        return CategoryInferenceResult(
            category="Audio",
            confidence=0.98,
            rationale=f"Audio file type .{asset.asset_type} is routed as Audio",
        )

    if asset.asset_type in {"mp4", "avi", "mov", "wmv", "mpeg", "mpg", "mkv", "flv", "webm", "3gp", "mts", "m2ts", "vob", "rmvb"}:
        return CategoryInferenceResult(
            category="Video",
            confidence=0.98,
            rationale=f"Video file type .{asset.asset_type} is routed as Video",
        )

    if asset.asset_type in {"csv", "tsv"}:
        return CategoryInferenceResult(
            category="Dataset",
            confidence=0.98,
            rationale=f"Delimited tabular file .{asset.asset_type} is routed as Dataset",
        )

    if asset.asset_type == "xlsx":
        sheet_count = int(asset.meta.get("sheet_count", 1))
        preview_rows = int(asset.meta.get("preview_rows", 0))
        max_columns = int(asset.meta.get("max_columns", 0))
        if sheet_count > 1 or preview_rows >= 6 or max_columns >= 4:
            return CategoryInferenceResult(
                category="Dataset",
                confidence=0.82,
                rationale="Spreadsheet shows tabular dataset signals such as multiple sheets, several rows, or multiple columns",
            )
        return CategoryInferenceResult(
            category="Document",
            confidence=0.58,
            rationale="Spreadsheet looks lightweight enough to remain document-like; category remains tentative",
        )

    return CategoryInferenceResult(
        category="Document",
        confidence=0.5,
        rationale=f"Fallback category routing applied for .{asset.asset_type}",
    )
