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
        
        # Run loading in a background thread to prevent blocking the FastAPI startup
        thread = threading.Thread(target=self._load_models_internal)
        thread.daemon = True
        thread.start()

    def _load_models_internal(self):
        print("Loading ML Models (Background Thread)...")
        try:
            if UnifiedDetector:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.detector = UnifiedDetector(device=device)
                self.detector.warmup() 
                self.risk_engine = RiskScoringEngine()
                self.anonymizer = PrivacyAnonymizer()
                self.loaded = True
                print("ML Models loaded successfully in background.")
            else:
                print("ML dependencies missing, running in mock mode.")
        except Exception as e:
            print(f"Error loading ML models: {e}")
            self.detector = None
            self.risk_engine = None
            self.anonymizer = None

ml_service = MLService.get_instance()
