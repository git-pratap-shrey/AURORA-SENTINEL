"""
Pytest Configuration and Shared Fixtures
"""
import pytest
import sys
import os
import numpy as np
import cv2
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session")
def fight_frame():
    """Generate synthetic fight frame with two overlapping persons"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    
    # Person 1: Aggressive stance
    cv2.rectangle(frame, (150, 150), (230, 350), (0, 0, 255), -1)
    cv2.rectangle(frame, (120, 170), (150, 210), (0, 0, 255), -1)  # Raised arm
    cv2.rectangle(frame, (230, 170), (260, 210), (0, 0, 255), -1)  # Raised arm
    
    # Person 2: Overlapping (grappling)
    cv2.rectangle(frame, (200, 160), (280, 360), (255, 0, 0), -1)
    cv2.rectangle(frame, (160, 240), (200, 260), (255, 0, 0), -1)  # Punch
    
    return frame

@pytest.fixture(scope="session")
def normal_frame():
    """Generate normal activity frame"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (60, 60, 60)
    
    # Person standing normally
    cv2.rectangle(frame, (280, 150), (360, 350), (0, 255, 0), -1)
    cv2.rectangle(frame, (270, 230), (280, 300), (0, 255, 0), -1)  # Arm at side
    cv2.rectangle(frame, (360, 230), (370, 300), (0, 255, 0), -1)  # Arm at side
    
    return frame

@pytest.fixture(scope="session")
def weapon_frame():
    """Generate frame with weapon"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    
    # Person holding weapon
    cv2.rectangle(frame, (250, 150), (330, 350), (0, 255, 255), -1)
    # Weapon (knife-like object)
    cv2.rectangle(frame, (330, 230), (400, 240), (200, 200, 200), -1)
    
    return frame

@pytest.fixture(scope="session")
def fight_video_path(fight_frame):
    """Generate temporary fight video file"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        video_path = tmp.name
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30, (640, 480))
    
    # Write 90 frames (3 seconds at 30fps)
    for i in range(90):
        out.write(fight_frame)
    
    out.release()
    
    yield video_path
    
    # Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)


@pytest.fixture(scope="session")
def app_client():
    """FastAPI TestClient with pre-loaded models"""
    try:
        from fastapi.testclient import TestClient
        from backend.api.main import app
        from backend.services.ml_service import ml_service
        
        # Load models synchronously
        print("Loading ML models for testing...")
        ml_service.load_models()
        
        client = TestClient(app)
        return client
    except Exception as e:
        pytest.skip(f"Could not create test client: {e}")

@pytest.fixture
def mock_vlm_responses():
    """Mock VLM provider responses for deterministic testing"""
    return {
        "fight_high_risk": {
            "provider": "test_mock",
            "description": "CRITICAL: Fight detected with aggressive postures and close combat",
            "risk_score": 95,
            "latency": 0.1,
            "details": {"confidence": 0.95, "scene_type": "real_fight"}
        },
        "boxing_low_risk": {
            "provider": "test_mock",
            "description": "Boxing match in ring with referee present, controlled sparring",
            "risk_score": 15,
            "latency": 0.1,
            "details": {"confidence": 0.85, "scene_type": "boxing"}
        },
        "normal_activity": {
            "provider": "test_mock",
            "description": "Normal activity, people standing and talking",
            "risk_score": 5,
            "latency": 0.1,
            "details": {"confidence": 0.90, "scene_type": "normal"}
        },
        "weapon_detected": {
            "provider": "test_mock",
            "description": "ALERT: Weapon detected in person's hand, potential threat",
            "risk_score": 90,
            "latency": 0.1,
            "details": {"confidence": 0.88, "scene_type": "weapon"}
        }
    }

@pytest.fixture
def sample_detection_data():
    """Sample detection data for risk engine testing"""
    return {
        "poses": [
            {
                "track_id": 1,
                "bbox": [150, 150, 230, 350],
                "keypoints": np.array([[190, 170], [190, 200], [190, 250], [190, 300]]),
                "confidence": np.array([0.9, 0.9, 0.9, 0.9]),
                "person_height": 200
            },
            {
                "track_id": 2,
                "bbox": [200, 160, 280, 360],
                "keypoints": np.array([[240, 180], [240, 210], [240, 260], [240, 310]]),
                "confidence": np.array([0.9, 0.9, 0.9, 0.9]),
                "person_height": 200
            }
        ],
        "objects": [
            {"class": "person", "confidence": 0.95, "bbox": [150, 150, 230, 350]},
            {"class": "person", "confidence": 0.93, "bbox": [200, 160, 280, 360]}
        ],
        "weapons": [],
        "timestamp": 1.0,
        "frame_number": 30
    }
