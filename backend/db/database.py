from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# DATABASE_URL from environment variable (e.g., PostgreSQL for production)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./aurora.db")

# SQLite specific argument: Increase timeout for Windows concurrency stability
connect_args = {"check_same_thread": False, "timeout": 30} if DATABASE_URL.startswith("sqlite") else {}

try:
    engine = create_engine(DATABASE_URL, connect_args=connect_args)
    # Test connection
    with engine.connect() as connection:
        pass
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    print(f"Connected to Database Service: {DATABASE_URL} (Timeout: 30s)")
except Exception as e:
    print(f"Database connection error: {e}. Falling back to SQLite for safety.")
    DATABASE_URL = "sqlite:///./aurora.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False, "timeout": 30})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()


def ensure_alert_columns():
    """
    Backward-compatible schema safety for legacy SQLite databases.
    Adds missing alert columns without requiring destructive migrations.
    """
    if not DATABASE_URL.startswith("sqlite"):
        return

    required_columns = {
        "ml_score": "FLOAT",
        "ai_score": "FLOAT",
        "final_score": "FLOAT",
        "detection_source": "VARCHAR",
        "ai_explanation": "VARCHAR",
        "ai_scene_type": "VARCHAR",
        "ai_confidence": "FLOAT",
    }

    with engine.begin() as conn:
        try:
            rows = conn.execute(text("PRAGMA table_info(alerts)")).fetchall()
        except Exception as e:
            print(f"Schema check skipped (alerts table unavailable): {e}")
            return

        existing = {row[1] for row in rows}
        for col_name, col_type in required_columns.items():
            if col_name in existing:
                continue
            try:
                conn.execute(text(f"ALTER TABLE alerts ADD COLUMN {col_name} {col_type}"))
                print(f"[DB] Added missing alerts column: {col_name}")
            except Exception as e:
                print(f"[DB] Failed to add alerts column {col_name}: {e}")
