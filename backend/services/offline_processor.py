import cv2
import os
import sys
import json
import time
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()

from backend.services.vlm_service import vlm_service

class OfflineProcessor:
    def __init__(self, storage_dir="storage/recordings", metadata_file="storage/metadata.json"):
        self.storage_dir = storage_dir
        self.metadata_file = metadata_file
        self.ensure_metadata_file()

    def ensure_metadata_file(self):
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w') as f:
                json.dump([], f)
            print(f"Created metadata registry: {self.metadata_file}")

    def load_metadata(self):
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def save_metadata(self, data):
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=4)

    def process_video(self, video_filename):
        video_path = os.path.join(self.storage_dir, video_filename)
        if not os.path.exists(video_path):
            print(f"Error: Video not found {video_path}")
            return

        print(f"Processing video: {video_filename}...")
        
        # Load existing metadata to check if already processed
        metadata_db = self.load_metadata()
        if any(item['filename'] == video_filename for item in metadata_db):
            print(f"Skipping {video_filename} (Already processed)")
            return

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video Info: {duration:.1f}s, {fps}fps")

        # Configuration
        ANALYSIS_INTERVAL = 5 # Analyze every 5 seconds
        frame_step = int(fps * ANALYSIS_INTERVAL)
        
        events = []
        current_frame = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame % frame_step == 0:
                timestamp = current_frame / fps
                print(f"  Analyzing frame at {timestamp:.1f}s...")
                
                # Convert to PIL/Bytes for VLM
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                # Prompt - STRICT JSON & THREAT DETECTION
                # Prompt - ENHANCED FORENSIC ANALYSIS
                prompt = (
                    "You are a critical surveillance AI. Analyze this scene for security threats. "
                    "Describe any fighting, weapons, aggression, or suspicious behavior immediately. "
                    "Do not use soft language or give benefit of doubt. If people are actually fighting, say 'Physical Altercation'.  "
                )
                # Prompt - DESCRIPTIVE NARRATIVE (No JSON)
                # prompt = (
                #     "Describe this surveillance scene in detail. "
                #     "Focus on people, clothing, actions, and any potential security threats (weapons, fighting, fire). "
                #     "Be objective and precise."
                # )

                # Call VLM
                result = vlm_service.analyze_scene(pil_img, prompt)
                description = result.get('description', '').strip()

                # Basic heuristic for metadata (to keep frontend working)
                detected_threats = []
                severity = "low"
                
                lower_desc = description.lower()
                threat_keywords = {
                    "gun": "gun", "firearm": "gun", "rifle": "gun", "pistol": "gun",
                    "knife": "knife", "blade": "knife", "machete": "knife",
                    "fire": "fire", "explosion": "fire",
                    "fighting": "fight", "punching": "fight", "kicking": "fight", "brawl": "fight",
                    "blood": "blood"
                }

                for k, v in threat_keywords.items():
                    if k in lower_desc:
                        if v not in detected_threats:
                            detected_threats.append(v)

                if len(detected_threats) > 0:
                    severity = "high"
                elif "aggressive" in lower_desc or "shouting" in lower_desc or "pushing" in lower_desc:
                    severity = "medium"

                # Save event
                event = {
                    "timestamp": round(timestamp, 2),
                    "description": description,
                    "threats": detected_threats,
                    "severity": severity,
                    "confidence": 0, # Placeholder
                    "provider": vlm_service.provider_name
                }
                
                # Store text for semantic search (combine fields)
                # We store the rich text description for searching, but keep structured data for UI
                events.append(event)
                
                print(f"  [Time: {round(timestamp, 2)}s] {event.get('description', 'Processed')[:50]}... | Severity: {severity}")
            
            current_frame += 1
            
        cap.release()
        
        # ---------------------------------------------------------------------
        # AUDIO ANALYSIS (New Phase)
        # ---------------------------------------------------------------------
        from backend.services.audio_service import audio_service
        
        print("  Starting Audio Analysis...")
        audio_events = audio_service.analyze_video(video_path)
        print(f"  Audio Analysis Complete. Found {len(audio_events)} events.")
        
        # Merge Video and Audio Events
        # We just append them. The frontend sorts by timestamp.
        all_events = events + audio_events
        
        # Sort by timestamp
        all_events.sort(key=lambda x: x['timestamp'])

        # Save results
        record = {
            "id": f"vid_{int(time.time())}",
            "filename": video_filename,
            "processed_at": datetime.now().isoformat(),
            "events": all_events
        }
        
        metadata_db.append(record)
        self.save_metadata(metadata_db)
        print(f"Finished processing {video_filename}. Metadata saved.")

    def scan_and_process(self):
        print(f"Scanning {self.storage_dir} for new videos...")
        if not os.path.exists(self.storage_dir):
            print(f"Storage directory {self.storage_dir} does not exist.")
            return

        files = [f for f in os.listdir(self.storage_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]
        
        if not files:
            print("No videos found.")
            return

        for video_file in files:
            self.process_video(video_file)

if __name__ == "__main__":
    processor = OfflineProcessor()
    processor.scan_and_process()
