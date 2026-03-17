"""
Generate 20 Normal CCTV Videos for Testing
"""
import cv2
import numpy as np
import os
import json
from datetime import datetime

def create_cctv_frame(frame_num, activity, total_frames):
    """Create realistic CCTV frame with normal activity"""
    # CCTV-style background (grayish, slightly grainy)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (50, 55, 50)  # Slightly greenish gray (typical CCTV)
    
    # Add CCTV grain/noise
    noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add timestamp overlay (typical CCTV feature)
    timestamp = f"2026-03-03 14:{30 + frame_num//25:02d}:{frame_num%25:02d}"
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    if activity == "walking":
        # Person walking across screen
        progress = frame_num / total_frames
        x = int(100 + progress * 400)
        y = 200
        
        # Person body
        cv2.rectangle(frame, (x, y), (x+60, y+180), (80, 90, 80), -1)
        # Arms swinging
        arm_swing = int(20 * np.sin(frame_num * 0.3))
        cv2.rectangle(frame, (x-15+arm_swing, y+60), (x, y+120), (80, 90, 80), -1)
        cv2.rectangle(frame, (x+60, y+60), (x+75-arm_swing, y+120), (80, 90, 80), -1)
        # Legs
        leg_swing = int(15 * np.sin(frame_num * 0.4))
        cv2.rectangle(frame, (x+10+leg_swing, y+180), (x+25, y+240), (70, 80, 70), -1)
        cv2.rectangle(frame, (x+35-leg_swing, y+180), (x+50, y+240), (70, 80, 70), -1)
        
    elif activity == "standing":
        # Person standing still
        x, y = 280, 150
        cv2.rectangle(frame, (x, y), (x+70, y+200), (85, 95, 85), -1)
        # Arms at sides
        cv2.rectangle(frame, (x-10, y+80), (x, y+150), (85, 95, 85), -1)
        cv2.rectangle(frame, (x+70, y+80), (x+80, y+150), (85, 95, 85), -1)
        # Slight movement (breathing, shifting weight)
        shift = int(2 * np.sin(frame_num * 0.1))
        frame = np.roll(frame, shift, axis=1)
        
    elif activity == "sitting":
        # Person sitting
        x, y = 250, 250
        cv2.rectangle(frame, (x, y), (x+80, y+120), (90, 100, 90), -1)
        # Arms on lap
        cv2.rectangle(frame, (x+10, y+60), (x+70, y+80), (90, 100, 90), -1)
        
    elif activity == "talking":
        # Two people standing and talking
        # Person 1
        x1, y1 = 200, 150
        cv2.rectangle(frame, (x1, y1), (x1+70, y1+200), (85, 95, 85), -1)
        cv2.rectangle(frame, (x1-10, y1+80), (x1, y1+150), (85, 95, 85), -1)
        cv2.rectangle(frame, (x1+70, y1+80), (x1+80, y1+150), (85, 95, 85), -1)
        
        # Person 2 (facing person 1)
        x2, y2 = 350, 150
        cv2.rectangle(frame, (x2, y2), (x2+70, y2+200), (80, 90, 80), -1)
        cv2.rectangle(frame, (x2-10, y2+80), (x2, y2+150), (80, 90, 80), -1)
        cv2.rectangle(frame, (x2+70, y2+80), (x2+80, y2+150), (80, 90, 80), -1)
        
        # Slight gesturing
        gesture = int(10 * np.sin(frame_num * 0.2))
        if gesture > 0:
            cv2.rectangle(frame, (x1+70, y1+60), (x1+90, y1+80), (85, 95, 85), -1)
    
    # Add slight motion blur for realism
    if frame_num % 3 == 0:
        kernel = np.ones((2, 2), np.float32) / 4
        frame = cv2.filter2D(frame, -1, kernel)
    
    return frame

def create_cctv_video(filename, activity, duration_seconds=5.0, fps=25):
    """Generate realistic CCTV normal activity video"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (640, 480))
    
    total_frames = int(duration_seconds * fps)
    
    for i in range(total_frames):
        frame = create_cctv_frame(i, activity, total_frames)
        out.write(frame)
    
    out.release()
    
    return {
        "filename": os.path.basename(filename),
        "has_fight": False,
        "has_weapon": False,
        "activity": activity,
        "num_persons": 1 if activity in ['walking', 'standing', 'sitting'] else 2,
        "duration": duration_seconds,
        "fps": fps,
        "expected_ml_score": "< 30",
        "expected_vlm_score": "< 30",
        "expected_alerts": "0"
    }

if __name__ == "__main__":
    output_dir = "tests/synthetic_data/videos"
    os.makedirs(output_dir, exist_ok=True)
    
    activities = ['walking', 'standing', 'sitting', 'talking']
    dataset = {
        "created_at": datetime.now().isoformat(),
        "type": "normal_cctv",
        "videos": []
    }
    
    print("=" * 60)
    print("GENERATING 20 NORMAL CCTV TEST VIDEOS")
    print("=" * 60)
    print()
    
    for i in range(20):
        activity = activities[i % len(activities)]
        filename = os.path.join(output_dir, f"normal_cctv_{i+1:02d}_{activity}.mp4")
        
        print(f"[{i+1}/20] Creating {os.path.basename(filename)}...")
        gt = create_cctv_video(filename, activity, duration_seconds=5.0)
        dataset["videos"].append({"path": filename, "ground_truth": gt})
    
    # Save dataset info
    info_path = os.path.join(output_dir, "normal_cctv_dataset.json")
    with open(info_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print()
    print("=" * 60)
    print("✅ GENERATION COMPLETE")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"  - Total videos: {len(dataset['videos'])}")
    print(f"  - Activities: walking, standing, sitting, talking")
    print(f"  - Duration: 5 seconds each @ 25fps")
    print(f"  - Expected ML score: < 30%")
    print(f"  - Expected VLM score: < 30%")
    print(f"  - Location: {output_dir}")
    print(f"  - Info file: {info_path}")
