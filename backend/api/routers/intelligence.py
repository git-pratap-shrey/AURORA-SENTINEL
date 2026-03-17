from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from backend.services.search_service import search_service
from backend.services.offline_processor import offline_processor
import os
import cv2
from PIL import Image
from backend.services.vlm_service import vlm_service

router = APIRouter()

class SearchQuery(BaseModel):
    query: str
    limit: int = 5

class SearchChatRequest(BaseModel):
    question: str
    filename: Optional[str] = None

class SearchResult(BaseModel):
    filename: str
    timestamp: float
    description: str
    score: float
    severity: str
    threats: List[str]
    provider: str
    confidence: float
    timestamp_seconds: Optional[float] = None

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
async def search_archive(q: str, limit: int = 5, filename: Optional[str] = None):
    """
    Semantic Search: "Find me a person with a knife"
    """
    try:
        results = search_service.search(q, limit, filename=filename)
        
        # Transform for frontend
        response = []
        for hit in results:
            meta = hit.get('metadata', {})
            threats_list = meta.get('threats', "").split(",") if meta.get('threats') else []
            response.append({
                "filename": meta.get('filename', 'unknown'),
                "timestamp": float(meta.get('timestamp', 0)),
                "description": hit['description'], # The text chunk
                "score": hit['score'],
                "severity": meta.get('severity', "low"),
                "threats": threats_list,
                "provider": meta.get('provider', "unknown"),
                "confidence": float(meta.get('confidence', 0))
            })
        return response
    except Exception as e:
        print(f"Search API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latest")
async def get_latest_insights():
    """
    Returns the most recent AI insights from metadata.json
    Shows recent videos with their summaries
    """
    try:
        data = offline_processor.load_metadata()
        
        if not data:
            return []
        
        # Sort videos by processed_at (most recent first)
        data.sort(key=lambda x: x.get('processed_at', ''), reverse=True)
        
        # Return recent videos with their main summary
        recent_videos = []
        for vid in data[:20]:  # Last 20 videos
            events = vid.get('events', [])
            
            # Get the main summary (first event is usually the overall summary)
            main_summary = "No description available"
            severity = "low"
            threats = []
            confidence = 0.0
            
            if events:
                main_event = events[0]
                main_summary = main_event.get('description', main_summary)
                severity = main_event.get('severity', severity)
                threats = main_event.get('threats', threats)
                confidence = main_event.get('confidence', confidence)
            
            recent_videos.append({
                "filename": vid.get('filename', 'unknown'),
                "processed_at": vid.get('processed_at', ''),
                "description": main_summary,
                "severity": severity,
                "threats": threats,
                "confidence": confidence,
                "provider": events[0].get('provider', 'unknown') if events else 'unknown',
                "timestamp": 0.0,  # Main summary is at start
                "event_count": len(events)
            })
        
        return recent_videos
    except Exception as e:
        print(f"Error fetching latest: {e}")
        import traceback
        traceback.print_exc()
        return []

@router.get("/recent")
async def get_recent_videos():
    """
    Returns all recent videos with their AI summaries
    Alias for /latest for compatibility
    """
    return await get_latest_insights()
@router.post("/chat")
async def intelligence_chat(req: SearchChatRequest):
    """
    Smart conversational chat about videos using local AI.
    Supports follow-up questions and context-aware responses.
    """
    try:
        print(f"[CHAT] Question: {req.question}, filename: {req.filename}")
        
        # 1. Get the video file
        if not req.filename:
            try:
                data = offline_processor.load_metadata()
                if data and len(data) > 0:
                    latest = max(data, key=lambda x: x.get('processed_at', ''))
                    req.filename = latest.get('filename')
                    print(f"[CHAT] Using latest video: {req.filename}")
            except Exception as e:
                print(f"[CHAT] Could not find latest video: {e}")
        
        # 2. Extract frame from video
        image_data = None
        if req.filename:
            try:
                storage_dirs = [
                    os.getenv("STORAGE_DIR", "storage/clips"),
                    "storage/recordings",
                    "storage/processed",
                    "storage/temp",
                    "storage/bin"
                ]
                
                video_path = None
                for storage_dir in storage_dirs:
                    test_path = os.path.join(storage_dir, req.filename)
                    if os.path.exists(test_path):
                        video_path = test_path
                        break
                
                if video_path and os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        import base64
                        from io import BytesIO
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb_frame)
                        buffer = BytesIO()
                        pil_img.save(buffer, format='JPEG')
                        image_data = base64.b64encode(buffer.getvalue()).decode()
                        image_data = f"data:image/jpeg;base64,{image_data}"
                        print(f"[CHAT] Frame extracted successfully")
            except Exception as e:
                print(f"[CHAT] Frame extraction error: {e}")
        
        # 3. Use VLM service for smart Q&A (FREE - uses Ollama locally)
        if image_data:
            try:
                print(f"[CHAT] Using VLM service for question answering...")
                
                # Call VLM service's answer_question method
                result = await vlm_service.answer_question(image_data, req.question)
                
                if result and result.get('answer'):
                    return {
                        "answer": result['answer'],
                        "confidence": result.get('confidence', 0.7),
                        "provider": result.get('provider', 'vlm'),
                        "source": "visual_qa",
                        "filename": req.filename
                    }
            except Exception as e:
                print(f"[CHAT] VLM service error: {e}")
                import traceback
                traceback.print_exc()
        
        # 4. Fallback: Use metadata for context
        try:
            data = offline_processor.load_metadata()
            video_metadata = None
            
            if req.filename:
                for vid in data:
                    if vid.get('filename') == req.filename:
                        video_metadata = vid
                        break
            elif data:
                video_metadata = max(data, key=lambda x: x.get('processed_at', ''))
            
            if video_metadata:
                events = video_metadata.get('events', [])
                if events:
                    main_event = events[0]
                    description = main_event.get('description', '')
                    
                    # Smart answer based on question type
                    question_lower = req.question.lower()
                    
                    if any(word in question_lower for word in ['what', 'describe', 'see', 'happening']):
                        answer = description
                    elif 'boxing' in question_lower or 'sport' in question_lower:
                        if any(word in description.lower() for word in ['boxing', 'sparring', 'sport', 'training']):
                            answer = "Yes, this appears to be boxing or organized sport based on the analysis."
                        else:
                            answer = "No, this does not appear to be organized sport. " + description
                    elif 'fight' in question_lower or 'violence' in question_lower:
                        if any(word in description.lower() for word in ['fight', 'violence', 'aggression', 'assault']):
                            answer = "Yes, there appears to be fighting or violence. " + description
                        else:
                            answer = "No clear signs of fighting detected. " + description
                    elif 'how many' in question_lower or 'count' in question_lower:
                        answer = f"Based on the analysis: {description}"
                    else:
                        answer = description
                    
                    return {
                        "answer": answer,
                        "confidence": main_event.get('confidence', 0.6),
                        "provider": "metadata",
                        "source": "metadata_qa",
                        "filename": video_metadata.get('filename')
                    }
        except Exception as e:
            print(f"[CHAT] Metadata fallback error: {e}")
        
        # 5. Final fallback
        return {
            "answer": "Please upload a video first, then I can answer questions about it.",
            "confidence": 0.0,
            "provider": "none",
            "source": "no_data",
            "filename": req.filename
        }
    except Exception as e:
        print(f"[CHAT] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
