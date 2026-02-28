from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from backend.services.search.search_service import SearchService
from backend.services.perception.offline_processor import OfflineProcessor
import os

router = APIRouter()
search_service = SearchService() # Singleton
offline_processor = OfflineProcessor()

class SearchQuery(BaseModel):
    query: str
    limit: int = 5

class SearchResult(BaseModel):
    filename: str
    timestamp: float
    description: str
    score: float
    severity: str
    threats: List[str]
    provider: str
    confidence: float

@router.post("/index")
async def trigger_indexing(background_tasks: BackgroundTasks):
    """
    Scans metadata.json and updates the Vector DB.
    """
    try:
        # Run in background to avoid blocking
        background_tasks.add_task(search_service.index_metadata)
        return {"status": "Indexing started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process")
async def trigger_processing(background_tasks: BackgroundTasks):
    """
    Scans storage/recordings and runs VLM analysis on new videos.
    """
    try:
        background_tasks.add_task(offline_processor.scan_and_process)
        return {"status": "Offline Processing started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search", response_model=List[SearchResult])
async def search_archive(q: str, limit: int = 5):
    """
    Semantic Search: "Find me a person with a knife"
    """
    try:
        results = search_service.search(q, limit)
        
        # Transform for frontend
        response = []
        for hit in results:
            threats_list = hit['metadata'].get('threats', "").split(",") if hit['metadata'].get('threats') else []
            response.append({
                "filename": hit['metadata']['filename'],
                "timestamp": float(hit['metadata']['timestamp']),
                "description": hit['description'], # The text chunk
                "score": hit['score'],
                "severity": hit['metadata'].get('severity', "low"),
                "threats": threats_list,
                "provider": hit['metadata'].get('provider', "unknown"),
                "confidence": float(hit['metadata'].get('confidence', 0))
            })
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latest")
async def get_latest_insights():
    """
    Returns the most recent AI insights from metadata.json
    """
    try:
        data = offline_processor.load_metadata()
        # Flatten events
        all_events = []
        for vid in data:
            for evt in vid['events']:
                all_events.append({
                    "filename": vid['filename'],
                    "timestamp": evt['timestamp'],
                    "description": evt['description'],
                    "processed_at": vid['processed_at'],
                    "severity": evt.get('severity', 'low'),
                    "threats": evt.get('threats', []),
                    "provider": evt.get('provider', 'unknown'),
                    "confidence": evt.get('confidence', 0)
                })
        
        # Sort by processed_at (descending)
        all_events.sort(key=lambda x: x['processed_at'], reverse=True)
        return all_events[:20]
    except Exception as e:
        print(f"Error fetching latest: {e}")
        return []
