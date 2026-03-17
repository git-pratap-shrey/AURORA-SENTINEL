"""
Synthetic Video Generator for Testing
Creates perfect ground truth videos for ML/VLM testing
"""
import cv2
import numpy as np
import os
import json
from datetime import datetime

class SyntheticVideoGenerator:
    def __init__(self, output_dir="tests/synthetic_data/videos"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def create_fight_frame(self, frame_num=0, intensity='high'):
        """Generate single frame with fighting poses"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)  # Dark background
        
        # Person 1: Aggressive stance (left side)
        x1, y1 = 150, 150
        cv2.rectangle(frame, (x1, y1), (x1+80, y1+200), (0, 0, 255), -1)  # Red person
        # Raised arms (fighting pose)
        cv2.rectangle(frame, (x1-30, y1+20), (x1, y1+60), (0, 0, 255), -1)  # Left arm up
        cv2.rectangle(frame, (x1+80, y1+20), (x1+110, y1+60), (0, 0, 255), -1)  # Right arm up
        
        # Person 2: Overlapping/grappling (right side)
        x2, y2 = 200, 160
        cv2.rectangle(frame, (x2, y2), (x2+80, y2+200), (255, 0, 0), -1)  # Blue person
        # Arms extended (striking pose)
        cv2.rectangle(frame, (x2-40, y2+80), (x2, y2+100), (255, 0, 0), -1)  # Punch
        
        # Add motion blur for realism
        if intensity == 'high':
            kernel = np.ones((5, 5), np.float32) / 25
            frame = cv2.filter2D(frame, -1, kernel)
        
        # Add noise
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        return frame
    
    def create_normal_frame(self, frame_num=0):
        """Generate frame with normal activity"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (60, 60, 60)  # Gray background
        
        # Person standing normally
        x, y = 280, 150
        cv2.rectangle(frame, (x, y), (x+80, y+200), (0, 255, 0), -1)  # Green person
        # Arms at sides
        cv2.rectangle(frame, (x-10, y+80), (x, y+150), (0, 255, 0), -1)
        cv2.rectangle(frame, (x+80, y+80), (x+90, y+150), (0, 255, 0), -1)
        
        return frame

    
    def create_weapon_frame(self, weapon_type='knife'):
        """Generate frame with weapon"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)
        
        # Person holding weapon
        x, y = 250, 150
        cv2.rectangle(frame, (x, y), (x+80, y+200), (0, 255, 255), -1)  # Cyan person
        
        # Weapon (elongated object in hand)
        if weapon_type == 'knife':
            cv2.rectangle(frame, (x+80, y+80), (x+150, y+90), (200, 200, 200), -1)
        elif weapon_type == 'gun':
            cv2.rectangle(frame, (x+80, y+85), (x+140, y+95), (100, 100, 100), -1)
        
        return frame
    
    def create_fight_video(self, filename="fight_test.mp4", duration_seconds=3.0, fps=30):
        """Generate complete fight video"""
        output_path = os.path.join(self.output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
        
        total_frames = int(duration_seconds * fps)
        
        for i in range(total_frames):
            frame = self.create_fight_frame(i, intensity='high')
            out.write(frame)
        
        out.release()
        
        # Generate ground truth
        ground_truth = {
            "filename": filename,
            "has_fight": True,
            "has_weapon": False,
            "num_persons": 2,
            "fight_intensity": "high",
            "duration": duration_seconds,
            "fps": fps,
            "expected_ml_score": ">= 70",
            "expected_alerts": ">= 1"
        }
        
        return output_path, ground_truth
    
    def create_normal_video(self, filename="normal_test.mp4", duration_seconds=3.0, fps=30):
        """Generate normal activity video"""
        output_path = os.path.join(self.output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
        
        total_frames = int(duration_seconds * fps)
        
        for i in range(total_frames):
            frame = self.create_normal_frame(i)
            out.write(frame)
        
        out.release()
        
        ground_truth = {
            "filename": filename,
            "has_fight": False,
            "has_weapon": False,
            "num_persons": 1,
            "activity": "standing",
            "duration": duration_seconds,
            "fps": fps,
            "expected_ml_score": "< 30",
            "expected_alerts": "0"
        }
        
        return output_path, ground_truth

    
    def create_weapon_video(self, filename="weapon_test.mp4", weapon_type='knife', duration_seconds=3.0, fps=30):
        """Generate video with weapon"""
        output_path = os.path.join(self.output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
        
        total_frames = int(duration_seconds * fps)
        
        for i in range(total_frames):
            frame = self.create_weapon_frame(weapon_type)
            out.write(frame)
        
        out.release()
        
        ground_truth = {
            "filename": filename,
            "has_fight": False,
            "has_weapon": True,
            "weapon_type": weapon_type,
            "num_persons": 1,
            "duration": duration_seconds,
            "fps": fps,
            "expected_ml_score": ">= 80",
            "expected_alerts": ">= 1"
        }
        
        return output_path, ground_truth
    
    def create_test_dataset(self, num_fight=5, num_normal=5, num_weapon=3):
        """Create complete test dataset"""
        dataset = {
            "created_at": datetime.now().isoformat(),
            "videos": []
        }
        
        print(f"Generating {num_fight} fight videos...")
        for i in range(num_fight):
            path, gt = self.create_fight_video(f"fight_{i+1}.mp4")
            dataset["videos"].append({"path": path, "ground_truth": gt})
        
        print(f"Generating {num_normal} normal videos...")
        for i in range(num_normal):
            path, gt = self.create_normal_video(f"normal_{i+1}.mp4")
            dataset["videos"].append({"path": path, "ground_truth": gt})
        
        print(f"Generating {num_weapon} weapon videos...")
        for i in range(num_weapon):
            weapon = ['knife', 'gun'][i % 2]
            path, gt = self.create_weapon_video(f"weapon_{i+1}.mp4", weapon)
            dataset["videos"].append({"path": path, "ground_truth": gt})
        
        # Save dataset info
        info_path = os.path.join(self.output_dir, "dataset_info.json")
        with open(info_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✅ Dataset created: {len(dataset['videos'])} videos")
        print(f"📁 Location: {self.output_dir}")
        print(f"📄 Info: {info_path}")
        
        return dataset

if __name__ == "__main__":
    generator = SyntheticVideoGenerator()
    dataset = generator.create_test_dataset(num_fight=5, num_normal=5, num_weapon=3)

    
    def create_cctv_normal_video(self, filename="normal_cctv_1.mp4", activity="walking", duration_seconds=5.0, fps=25):
        """Generate realistic CCTV normal activity video"""
        output_path = os.path.join(self.output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
        
        total_frames = int(duration_seconds * fps)
        
        for i in range(total_frames):
            frame = self._create_cctv_frame(i, activity, total_frames)
            out.write(frame)
        
        out.release()
        
        ground_truth = {
            "filename": filename,
            "has_fight": False,
            "has_weapon": False,
            "activity": activity,
            "num_persons": 1 if activity in ['walking', 'standing'] else 2,
            "duration": duration_seconds,
            "fps": fps,
            "expected_ml_score": "< 30",
            "expected_vlm_score": "< 30",
            "expected_alerts": "0"
        }
        
        return output_path, ground_truth
    
    def _create_cctv_frame(self, frame_num, activity, total_frames):
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
    
    def create_normal_cctv_dataset(self, num_videos=20):
        """Create dataset of normal CCTV videos"""
        activities = ['walking', 'standing', 'sitting', 'talking']
        dataset = {
            "created_at": datetime.now().isoformat(),
            "type": "normal_cctv",
            "videos": []
        }
        
        print(f"\n🎬 Generating {num_videos} normal CCTV videos...")
        
        for i in range(num_videos):
            activity = activities[i % len(activities)]
            filename = f"normal_cctv_{i+1:02d}_{activity}.mp4"
            
            print(f"  [{i+1}/{num_videos}] Creating {filename}...")
            path, gt = self.create_cctv_normal_video(filename, activity, duration_seconds=5.0)
            dataset["videos"].append({"path": path, "ground_truth": gt})
        
        # Save dataset info
        info_path = os.path.join(self.output_dir, "normal_cctv_dataset.json")
        with open(info_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n✅ Normal CCTV dataset created: {num_videos} videos")
        print(f"📁 Location: {self.output_dir}")
        print(f"📄 Info: {info_path}")
        
        return dataset

if __name__ == "__main__":
    generator = SyntheticVideoGenerator()
    
    # Generate normal CCTV videos
    print("=" * 60)
    print("GENERATING NORMAL CCTV TEST VIDEOS")
    print("=" * 60)
    dataset = generator.create_normal_cctv_dataset(num_videos=20)
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"  - Total videos: {len(dataset['videos'])}")
    print(f"  - Activities: walking, standing, sitting, talking")
    print(f"  - Duration: 5 seconds each @ 25fps")
    print(f"  - Expected ML score: < 30%")
    print(f"  - Expected VLM score: < 30%")
