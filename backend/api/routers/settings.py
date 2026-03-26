from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.db.database import SessionLocal
from backend.db.models import SystemSetting
from backend.api.deps import get_db
from pydantic import BaseModel

from backend.services.system_settings_service import (
    VLM_INTERVAL_KEY,
    get_vlm_interval_seconds,
    set_vlm_interval_seconds,
)

router = APIRouter()

class SettingUpdate(BaseModel):
    value: str


class VlmIntervalUpdate(BaseModel):
    seconds: int

@router.get("/maintenance")
async def get_maintenance_mode(db: Session = Depends(get_db)):
    """Get the current maintenance mode status"""
    setting = db.query(SystemSetting).filter(SystemSetting.key == "maintenance_mode").first()
    if not setting:
        return {"maintenance_mode": False}
    return {"maintenance_mode": setting.value.lower() == "true"}

@router.post("/maintenance")
async def set_maintenance_mode(req: SettingUpdate, db: Session = Depends(get_db)):
    """Set the maintenance mode status"""
    setting = db.query(SystemSetting).filter(SystemSetting.key == "maintenance_mode").first()
    if not setting:
        setting = SystemSetting(key="maintenance_mode", value=req.value)
        db.add(setting)
    else:
        setting.value = req.value
    
    db.commit()
    return {"status": "success", "maintenance_mode": setting.value.lower() == "true"}


@router.get("/vlm-interval")
async def get_vlm_interval():
    """Get persisted global VLM analysis interval in seconds."""
    interval = get_vlm_interval_seconds(default_value=10)
    return {"key": VLM_INTERVAL_KEY, "seconds": int(interval)}


@router.post("/vlm-interval")
async def set_vlm_interval(req: VlmIntervalUpdate):
    """
    Set persisted global VLM interval.
    Guardrails: 2-30 seconds.
    """
    seconds = int(req.seconds)
    if seconds < 2 or seconds > 30:
        raise HTTPException(status_code=400, detail="seconds must be between 2 and 30")

    saved = set_vlm_interval_seconds(seconds)
    return {"status": "success", "key": VLM_INTERVAL_KEY, "seconds": int(saved)}
