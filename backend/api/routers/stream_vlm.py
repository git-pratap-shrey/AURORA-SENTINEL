from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.services.ml_service import ml_service
from backend.services.video_storage_service import video_storage_service
from backend.services.vlm_service import vlm_service # NEW
from backend.db.database import SessionLocal
from backend.db.models import Alert
import cv2
import numpy as np
import asyncio
import base64
from datetime import datetime
import time
from PIL import Image

router = APIRouter()

# Helper for non-blocking DB write
def save_alert_sync(alert_data):
    try:
        db = SessionLocal()
        new_alert = Alert(
            level=alert_data['level'],
            risk_score=float(alert_data['score']),
            camera_id="CAM-01",
            location="Main Feed",
            risk_factors=alert_data.get('top_factors', []),
            status="pending",
            timestamp=datetime.utcnow()
        )
        db.add(new_alert)
        db.commit()
        db.refresh(new_alert)
        db.close()
        return new_alert.id
    except Exception as e:
        print(f"DB Write Error: {e}")
        return None

@router.websocket("/vlm-feed") # NEW ENDPOINT
async def websocket_vlm_feed(websocket: WebSocket):
    """
    VLM-Enhanced WebSocket endpoint
    Integrates Gemini/Ollama analysis into the loop.
    """
    await websocket.accept()
    print("WebSocket connected (VLM Mode)")
    
    frame_count = 0
    SKIP_FRAMES = 1
    
    # State
    last_alert_time = 0
    ALERT_COOLDOWN = 10
    
    # VLM State
    last_vlm_time = 0
    VLM_INTERVAL = 10 # seconds (Analyze every 5s by default)
    current_narrative = "Initializing AI Analysis..."
    vlm_task = None # Handle for the background task
    
    cached_result = {
        "detection": {"poses": [], "objects": []}, 
        "risk_score": 0, 
        "alert": None,
        "faces": []
    }

    try:
        while True:
            try:
                data = await websocket.receive_bytes()
            except Exception:
                break

            if not ml_service.detector:
                await websocket.send_json({"error": "Models Loading..."})
                await asyncio.sleep(0.5)
                continue

            # Decode
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None: continue
            
            start_proc = time.time()
            
            if frame_count % (SKIP_FRAMES + 1) == 0:
                try:
                    # 1. Check for completed VLM task (Non-Blocking Check)
                    if vlm_task and vlm_task.done():
                        try:
                            vlm_result = vlm_task.result()
                            print(f"Received VLM Result: {vlm_result.get('description', 'Error')[:50]}...")
                            current_narrative = vlm_result.get("description", "Analysis Failed")
                        except Exception as e:
                            print(f"VLM Task Error: {e}")
                        finally:
                            vlm_task = None # Reset for next run

                    is_vlm_running = (vlm_task is not None)

                    if is_vlm_running:
                        # OPTIMIZATION: System is busy with AI (16s+ load).
                        # Skip heavy YOLO inference to prevent video freeze.
                        # Reuse last known detection.
                        detection = cached_result["detection"]
                        risk_score = cached_result["risk_score"]
                        # Optional: Add visual indicator
                        current_narrative = "AI Thinking... (Video smooth)" 
                    else:
                        # Normal Mode: Run YOLO High FPS
                        detection = ml_service.detector.process_frame(frame)
                        risk_score, risk_factors = ml_service.risk_engine.calculate_risk(detection)
                        risk_score = risk_score or 0
                        
                        # 3. Trigger New VLM Analysis (If idle)
                        now = time.time()
                        should_trigger = (risk_score > 60 and (now - last_vlm_time > 3)) or (now - last_vlm_time > VLM_INTERVAL)
                        
                        if should_trigger:
                            print(f"Triggering VLM Analysis (Interval: {now - last_vlm_time:.1f}s)...")
                            
                            # Prepare args (Copy frame to avoid race conditions)
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(rgb_frame)
                            prompt = "Analyze this surveillance frame. Describe any potential threats, weapons, or aggressive behavior concisely. If safe, say 'Situation normal'."
                            
                            # Launch Background Task (Non-Blocking)
                            loop = asyncio.get_event_loop()
                            vlm_task = loop.create_task(
                                loop.run_in_executor(None, vlm_service.analyze_scene, pil_img, prompt)
                            )
                            last_vlm_time = now
                            current_narrative = "Analyzing Scene..." 

                    # Alert Logic
                    alert = None
                    if risk_score > 65:
                        alert = ml_service.risk_engine.generate_alert(risk_score, risk_factors if not is_vlm_running else [])
                        alert['level'] = alert['level'].upper()
                        # Add VLM insight to alert
                        alert['ai_analysis'] = current_narrative
                        
                        if risk_score > 80:
                            video_storage_service.start_recording("CAM-01")
                        
                        now_ts = datetime.utcnow().timestamp()
                        if alert and (now_ts - last_alert_time > ALERT_COOLDOWN):
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, save_alert_sync, alert)
                            last_alert_time = now_ts

                    video_storage_service.add_frame("CAM-01", frame)

                    cached_result.update({
                        "detection": detection,
                        "risk_score": risk_score,
                        "alert": alert
                    })
                except Exception as e:
                    print(f"ML Processing Failed: {e}")
            else:
                detection = cached_result["detection"]
                risk_score = cached_result["risk_score"]
                alert = cached_result["alert"]

            # Adapt Skip
            if time.time() - start_proc > 0.05:
                SKIP_FRAMES = 1
            else:
                SKIP_FRAMES = 0

            frame_count += 1
            
            # Anonymization (Reuse existing logic)
            anon_frame = frame # distinct VLM mode might skip this for raw analysis, but keeping for display
            if ml_service.anonymizer:
                try:
                     anon_frame = ml_service.anonymizer.anonymize_frame(
                        frame, detection.get('poses', []), mode='blur'
                    )
                except: pass

            # Draw Overlays (Simplified for VLM Mode)
            if detection:
                for obj in detection.get('objects', []):
                    x1, y1, x2, y2 = map(int, obj['bbox'])
                    cv2.rectangle(anon_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Encode
            _, buffer = cv2.imencode('.jpg', anon_frame)
            
            # Send Data
            try:
                if frame_count % 3 == 0 or alert:
                    await websocket.send_json({
                        "risk_score": risk_score,
                        "vlm_narrative": current_narrative, # NEW FIELD
                        "alert": alert,
                        "provider": vlm_service.provider_name
                    })
                await websocket.send_bytes(buffer.tobytes())
            except Exception:
                break

    except WebSocketDisconnect:
        print("VLM WebSocket disconnected")
    finally:
        try: await websocket.close()
        except: pass
