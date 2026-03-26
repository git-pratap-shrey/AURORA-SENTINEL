from typing import Optional

from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.db.models import SystemSetting


VLM_INTERVAL_KEY = "vlm_interval_seconds"


def _get_value(db: Session, key: str) -> Optional[str]:
    rec = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    return rec.value if rec else None


def _set_value(db: Session, key: str, value: str) -> None:
    rec = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    if rec is None:
        rec = SystemSetting(key=key, value=value)
        db.add(rec)
    else:
        rec.value = value
    db.commit()


def get_vlm_interval_seconds(default_value: int = 10) -> int:
    db = SessionLocal()
    try:
        raw = _get_value(db, VLM_INTERVAL_KEY)
        if raw is None:
            return int(default_value)
        parsed = int(float(raw))
        return parsed
    except Exception:
        return int(default_value)
    finally:
        db.close()


def set_vlm_interval_seconds(value: int) -> int:
    db = SessionLocal()
    try:
        normalized = int(value)
        _set_value(db, VLM_INTERVAL_KEY, str(normalized))
        return normalized
    finally:
        db.close()
