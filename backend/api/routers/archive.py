from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
from datetime import datetime

router = APIRouter()

# Configurable Storage Paths
STORAGE_PATH = os.path.abspath(os.getenv("CLIPS_PATH", "storage/clips"))
BIN_PATH = os.path.abspath(os.getenv("BIN_PATH", "storage/bin"))
PROCESSED_PATH = os.path.abspath(os.getenv("PROCESSED_PATH", "storage/processed"))

os.makedirs(STORAGE_PATH, exist_ok=True)
os.makedirs(BIN_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)


def _resolve_source_path(source: str) -> str:
    if source == "active":
        return STORAGE_PATH
    if source == "bin":
        return BIN_PATH
    if source == "processed":
        return PROCESSED_PATH
    return STORAGE_PATH


def _safe_file_path(root_path: str, filename: str) -> str:
    # Keep lookups strictly inside selected storage root.
    sanitized = os.path.basename(filename)
    candidate = os.path.abspath(os.path.join(root_path, sanitized))
    if os.path.commonpath([root_path, candidate]) != root_path:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return candidate

@router.get("/list")
async def list_archives(source: str = "active"):
    path = _resolve_source_path(source)
    
    if not os.path.exists(path):
        return {"clips": []}
    
    files = []
    for f in os.listdir(path):
        if f.endswith(".mp4"):
            f_path = os.path.join(path, f)
            stat = os.stat(f_path)
            files.append({
                "id": f,
                "name": f,
                "size": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "url": f"/archive/download/{f}?source={source}"
            })
    
    # Sort by newest first
    files.sort(key=lambda x: x["created_at"], reverse=True)
    return {"clips": files}

@router.get("/download/{filename}")
async def download_clip(filename: str, source: str = "active"):
    path = _resolve_source_path(source)
    file_path = _safe_file_path(path, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Clip not found")
        
    return FileResponse(file_path, media_type="video/mp4")


@router.post("/restore/{filename}")
async def restore_clip(filename: str):
    source_file = _safe_file_path(BIN_PATH, filename)
    if not os.path.exists(source_file):
        raise HTTPException(status_code=404, detail="Clip not found in bin")

    destination = _safe_file_path(STORAGE_PATH, filename)
    shutil.move(source_file, destination)
    return {"status": "ok", "message": "Clip restored to active storage", "name": os.path.basename(filename)}


@router.delete("/delete/{filename}")
async def delete_clip(filename: str, source: str = "bin"):
    if source not in {"active", "bin", "processed"}:
        raise HTTPException(status_code=400, detail="Invalid source")

    root_path = _resolve_source_path(source)
    file_path = _safe_file_path(root_path, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Clip not found")

    os.remove(file_path)
    return {"status": "ok", "message": "Clip deleted", "name": os.path.basename(filename), "source": source}
