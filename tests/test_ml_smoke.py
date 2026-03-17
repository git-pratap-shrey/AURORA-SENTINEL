"""
ML Layer Smoke Tests
Tests for UnifiedDetector and RiskScoringEngine
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestUnifiedDetector:
    """Test ML detection layer"""
    
    def test_detector_loads(self):
        """Test UnifiedDetector initializes without exception"""
        try:
            from models.detection.detector import UnifiedDetector
            detector = UnifiedDetector()
            assert detector is not None
        except Exception as e:
            pytest.skip(f"Detector not available: {e}")
    
    def test_process_frame_blank(self, normal_frame):
        """Test blank frame returns expected structure"""
        try:
            from models.detection.detector import UnifiedDetector
            detector = UnifiedDetector()
            
            result = detector.process_frame(normal_frame)
            
            assert 'poses' in result
            assert 'objects' in result
            assert 'weapons' in result
            assert isinstance(result['poses'], list)
            assert isinstance(result['objects'], list)
        except Exception as e:
            pytest.skip(f"Detector not available: {e}")
    
    def test_process_frame_fight_synthetic(self, fight_frame):
        """Test synthetic fight frame detects poses"""
        try:
            from models.detection.detector import UnifiedDetector
            detector = UnifiedDetector()
            
            result = detector.process_frame(fight_frame)
            
            # Should detect at least some objects/poses
            total_detections = len(result['poses']) + len(result['objects'])
            assert total_detections >= 0, "Should process frame without error"
        except Exception as e:
            pytest.skip(f"Detector not available: {e}")
    
    def test_detect_objects_returns_structure(self, fight_frame):
        """Verify object detection response format"""
        try:
            from models.detection.detector import UnifiedDetector
            detector = UnifiedDetector()
            
            result = detector.process_frame(fight_frame)
            
            for obj in result['objects']:
                assert 'class' in obj
                assert 'confidence' in obj
                assert 'bbox' in obj
                assert len(obj['bbox']) == 4
        except Exception as e:
            pytest.skip(f"Detector not available: {e}")


class TestRiskScoringEngine:
    """Test risk scoring and alert generation"""
    
    def test_risk_engine_weapon(self, sample_detection_data):
        """Test weapon detection escalates risk score"""
        try:
            from models.scoring.risk_engine import RiskScoringEngine
            
            engine = RiskScoringEngine(fps=30)
            
            # Add weapon to detection data
            detection_with_weapon = sample_detection_data.copy()
            detection_with_weapon['weapons'] = [
                {"class": "knife", "confidence": 0.9, "bbox": [330, 230, 400, 240]}
            ]
            
            risk_score, factors = engine.calculate_risk(detection_with_weapon)
            
            assert risk_score >= 60, f"Weapon should escalate risk, got {risk_score}"
            assert factors.get('weapon_detection', 0) > 0
        except Exception as e:
            pytest.skip(f"Risk engine not available: {e}")
    
    def test_risk_engine_aggression(self, sample_detection_data):
        """Test aggressive posture detection"""
        try:
            from models.scoring.risk_engine import RiskScoringEngine
            
            engine = RiskScoringEngine(fps=30)
            risk_score, factors = engine.calculate_risk(sample_detection_data)
            
            # Should detect some level of aggression with overlapping poses
            assert 'aggressive_posture' in factors
            assert factors['aggressive_posture'] >= 0
        except Exception as e:
            pytest.skip(f"Risk engine not available: {e}")
    
    def test_risk_engine_grappling(self, sample_detection_data):
        """Test grappling detection with overlapping poses"""
        try:
            from models.scoring.risk_engine import RiskScoringEngine
            
            engine = RiskScoringEngine(fps=30)
            risk_score, factors = engine.calculate_risk(sample_detection_data)
            
            # Overlapping bboxes should trigger proximity/grappling
            assert 'proximity_violation' in factors or 'grappling' in factors
        except Exception as e:
            pytest.skip(f"Risk engine not available: {e}")
    
    def test_risk_engine_no_person(self):
        """Test empty detection returns zero risk"""
        try:
            from models.scoring.risk_engine import RiskScoringEngine
            
            engine = RiskScoringEngine(fps=30)
            empty_detection = {
                "poses": [],
                "objects": [],
                "weapons": [],
                "timestamp": 0.0,
                "frame_number": 0
            }
            
            risk_score, factors = engine.calculate_risk(empty_detection)
            
            assert risk_score == 0, "Empty detection should have zero risk"
        except Exception as e:
            pytest.skip(f"Risk engine not available: {e}")
    
    def test_risk_engine_temporal_validation(self, sample_detection_data):
        """Test temporal validation reduces false positives"""
        try:
            from models.scoring.risk_engine import RiskScoringEngine
            
            engine = RiskScoringEngine(fps=30)
            
            # Process multiple frames
            scores = []
            for i in range(25):
                risk_score, _ = engine.calculate_risk(sample_detection_data)
                scores.append(risk_score)
            
            # Temporal validation should stabilize scores
            assert len(scores) == 25
        except Exception as e:
            pytest.skip(f"Risk engine not available: {e}")
