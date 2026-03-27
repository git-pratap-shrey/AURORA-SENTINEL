import sys
import os

# Add root to sys.path to ensure models can be imported
sys.path.append(os.getcwd())

try:
    from models.detection.detector import UnifiedDetector
    from models.scoring.risk_engine import RiskScoringEngine
    from models.privacy.anonymizer import PrivacyAnonymizer
except ImportError:
    print("Warning: Could not import machine learning models.")
    UnifiedDetector = None
    RiskScoringEngine = None
    PrivacyAnonymizer = None

import threading # NEW

class MLService:
    _instance = None
    _lock = threading.Lock() # NEW
    
    def __init__(self):
        self.detector = None
        self.risk_engine = None
        self.anonymizer = None
        self.loaded = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_models(self):
        if self.loaded:
            return
        # Load synchronously — must complete before accepting WebSocket connections
        self._load_models_internal()

    def load_models_async(self):
        """Non-blocking load for cases where background loading is acceptable."""
        if self.loaded:
            return
        thread = threading.Thread(target=self._load_models_internal)
        thread.daemon = True
        thread.start()

    def _load_models_internal(self):
        print("=" * 50)
        print("Loading ML Models...")
        try:
            if UnifiedDetector:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"  Device: {device.upper()}")
                print("  Loading YOLO detector...")
                self.detector = UnifiedDetector(device=device)
                print("  Warming up models...")
                self.detector.warmup()
                print("  Loading Risk Engine...")
                self.risk_engine = RiskScoringEngine()
                print("  Loading Anonymizer...")
                self.anonymizer = PrivacyAnonymizer()
                self.loaded = True
                print("ML Models loaded successfully.")
                print("=" * 50)
            else:
                print("WARNING: ML dependencies missing — running in mock mode (0% scores expected).")
                print("Install: pip install ultralytics torch")
                print("=" * 50)
        except Exception as e:
            import traceback
            print(f"ERROR loading ML models: {e}")
            traceback.print_exc()
            self.detector = None
            self.risk_engine = None
            self.anonymizer = None
            print("=" * 50)

ml_service = MLService.get_instance()
