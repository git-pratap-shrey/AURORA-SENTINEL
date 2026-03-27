import os
import cv2
import time
import shutil
from datetime import datetime, timedelta
import threading

# 3 dirname calls: video_storage_service.py → services/ → backend/ → project root
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

class Recording:
    def __init__(self, writer, path, start_time):
        self.writer = writer
        self.path = path
        self.start_time = start_time
        self.last_frame_time = start_time

class VideoStorageService:
    def __init__(self, base_path=None):
        # Default to project root's storage directory
        if base_path is None:
            base_path = os.path.join(_PROJECT_ROOT, "storage")
        self.base_path = base_path
        self.clips_path = os.path.join(base_path, "clips")
        self.bin_path = os.path.join(base_path, "bin")
        self.live_retention_hours = int(os.getenv("LIVE_CLIP_RETENTION_HOURS", "24"))
        self.bin_retention_days = int(os.getenv("BIN_RETENTION_DAYS", "7"))
        
        # Ensure directories exist
        os.makedirs(self.clips_path, exist_ok=True)
        os.makedirs(self.bin_path, exist_ok=True)
        
        self.active_recordings = {} # {camera_id: VideoWriter}
        self.recordings_lock = threading.Lock()
        self.cleanup_interval = 3600 # 1 hour
        self._start_cleanup_thread()

    def start_recording(self, camera_id, frame_size=(640, 480)):
        with self.recordings_lock:
            if camera_id in self.active_recordings:
                return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{camera_id}_{timestamp}.mp4"
        filepath = os.path.join(self.clips_path, filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filepath, fourcc, 10.0, frame_size)

        with self.recordings_lock:
            self.active_recordings[camera_id] = Recording(writer, filepath, time.time())
        print(f"Started smart recording for {camera_id}: {filename}")

    def add_frame(self, camera_id, frame):
        with self.recordings_lock:
            recording = self.active_recordings.get(camera_id)
        if recording is None:
            return

        recording.writer.write(frame)
        recording.last_frame_time = time.time()
        
        # Auto-stop after 30 seconds of recording to chunk files
        if time.time() - recording.start_time > 30:
            self.stop_recording(camera_id)

    def stop_recording(self, camera_id):
        with self.recordings_lock:
            recording = self.active_recordings.pop(camera_id, None)
        if recording is None:
            return

        recording.writer.release()
        print(f"Stopped recording for {camera_id}")

    def is_recording(self, camera_id):
        with self.recordings_lock:
            return camera_id in self.active_recordings

    def stop_all_recordings(self):
        with self.recordings_lock:
            camera_ids = list(self.active_recordings.keys())
        for camera_id in camera_ids:
            self.stop_recording(camera_id)

    def _start_cleanup_thread(self):
        thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        thread.start()

    def _cleanup_loop(self):
        while True:
            try:
                self.run_smart_cleanup()
            except Exception as e:
                # Keep daemon alive on transient filesystem/runtime errors.
                print(f"Smart cleanup error: {e}")
            time.sleep(self.cleanup_interval)

    def run_smart_cleanup(self):
        """
        Industry standard cleanup:
        - Move files older than LIVE_CLIP_RETENTION_HOURS to Bin.
        - Delete files from Bin older than BIN_RETENTION_DAYS.
        """
        now = datetime.now()
        
        # 1. Clips -> Bin
        for f in os.listdir(self.clips_path):
            file_path = os.path.join(self.clips_path, f)
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            if now - mtime > timedelta(hours=self.live_retention_hours):
                print(f"Moving {f} to Smart Bin")
                shutil.move(file_path, os.path.join(self.bin_path, f))
        
        # 2. Bin -> Trash
        for f in os.listdir(self.bin_path):
            file_path = os.path.join(self.bin_path, f)
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            if now - mtime > timedelta(days=self.bin_retention_days):
                print(f"Deleting expired clip from bin: {f}")
                os.remove(file_path)

video_storage_service = VideoStorageService()
