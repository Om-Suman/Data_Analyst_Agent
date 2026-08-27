from __future__ import annotations

import io
from typing import List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from backend.api.deps import get_current_session
from backend.session.state import SessionState
from backend.schemas.dataset import (
    DatasetItem,
    DatasetListResponse,
    DatasetPreviewResponse,
    SampleDatasetRequest,
    SetActiveRequest,
)
from backend.services.dataset_service import (
    get_dataset_preview,
    load_sample_dataset,
    process_and_register_file,
)

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("", response_model=DatasetListResponse)
def list_datasets(session: SessionState = Depends(get_current_session)):
    items = []
    for name, rec in session.datasets.items():
        uploaded_ts = rec["uploaded_at"].isoformat() if hasattr(rec.get("uploaded_at"), "isoformat") else str(rec.get("uploaded_at"))
        items.append(
            DatasetItem(
                name=name,
                source=rec.get("source", "upload"),
                uploaded_at=uploaded_ts,
                rows=int(rec.get("rows", 0)),
                cols=int(rec.get("cols", 0)),
                version=int(rec.get("version", 1)),
                is_active=(name == session.active_dataset),
                is_text=bool(rec.get("text_content")),
            )
        )
    return DatasetListResponse(datasets=items, active_dataset=session.active_dataset)


@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    session: SessionState = Depends(get_current_session),
):
    results = []
    errors = []

    for file in files:
        try:
            content = await file.read()
            res = process_and_register_file(session, file.filename, content)
            results.append(res)
        except Exception as exc:
            errors.append({"filename": file.filename, "error": str(exc)})

    if not results and errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process uploaded file(s): {errors[0]['error']}",
        )

    return {
        "success": True,
        "processed": results,
        "errors": errors,
        "active_dataset": session.active_dataset,
    }


@router.post("/sample")
def load_sample(
    req: SampleDatasetRequest,
    session: SessionState = Depends(get_current_session),
):
    try:
        res = load_sample_dataset(session, req.sample_name)
        return {
            "success": True,
            "sample": res,
            "active_dataset": session.active_dataset,
        }
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/active")
def set_active_dataset(
    req: SetActiveRequest,
    session: SessionState = Depends(get_current_session),
):
    if req.name not in session.datasets:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{req.name}' not found.")
    session.active_dataset = req.name
    return {"status": "ok", "active_dataset": req.name}


@router.delete("/{name}")
def delete_dataset(
    name: str,
    session: SessionState = Depends(get_current_session),
):
    if name not in session.datasets:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{name}' not found.")
    del session.datasets[name]
    if name in session.dataset_versions:
        del session.dataset_versions[name]
    if name in session.document_indexes:
        del session.document_indexes[name]

    if session.active_dataset == name:
        session.active_dataset = list(session.datasets.keys())[0] if session.datasets else None

    return {"status": "ok", "active_dataset": session.active_dataset}


@router.get("/preview", response_model=DatasetPreviewResponse)
def get_preview(
    limit: int = 50,
    session: SessionState = Depends(get_current_session),
):
    try:
        return get_dataset_preview(session, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/download/csv")
def download_csv(session: SessionState = Depends(get_current_session)):
    df = session.get_active_df()
    name = session.active_dataset or "dataset"
    if df is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active dataset.")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}.csv"'},
    )


@router.get("/download/excel")
def download_excel(session: SessionState = Depends(get_current_session)):
    df = session.get_active_df()
    name = session.active_dataset or "dataset"
    if df is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active dataset.")
    output = io.BytesIO()
    with pd_excel_writer(output) as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return Response(
        content=output.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{name}.xlsx"'},
    )


def pd_excel_writer(buffer):
    import pandas as pd
    return pd.ExcelWriter(buffer, engine="openpyxl")
