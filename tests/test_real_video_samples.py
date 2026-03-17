"""
Integration tests with real video samples.

Tests:
- Fight videos → expect scores 75-95, scene type real_fight
- Boxing videos → expect scores 20-35 (capped), scene type organized_sport
- Normal videos → expect scores 10-25, scene type normal
- Suspicious behavior → expect scores 60-75, scene type suspicious
- Verify no scores cluster around 40
- Verify no prank/drama classifications appear

Requirements: 1.2, 1.3, 1.4, 2.3, 2.4, 2.5, 2.6, 2.7
"""

import pytest
import sys
import os
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestRealVideoSamples:
    """Test with real video samples to validate scoring accuracy."""
    
    @pytest.fixture
    def ai_router_available(self):
        """Check if AI router is available."""
        try:
            sys.path.insert(0, 'ai-intelligence-layer')
            from aiRouter_enhanced import analyze_image
            return True
        except ImportError:
            return False
    
    def extract_frame(self, video_path, frame_number=30):
        """Extract a frame from video for testing."""
        if not os.path.exists(video_path):
            return None
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Convert to base64
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        buffer = BytesIO()
        pil_img.save(buffer, format='JPEG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{image_data}"
    
    @pytest.mark.skipif(not os.path.exists('ai-intelligence-layer/aiRouter_enhanced.py'),
                       reason="AI router not available")
    def test_fight_video_classification(self, ai_router_available):
        """
        Test fight videos → expect scores 75-95, scene type real_fight.
        Requirements: 1.2, 1.3, 2.3, 2.4
        """
        if not ai_router_available:
            pytest.skip("AI router not available")
        
        from aiRouter_enhanced import analyze_image
        
        # Look for fight video samples
        test_video_paths = [
            'storage/test_videos/fight_sample.mp4',
            'storage/recordings/fight_test.mp4',
            'tests/fixtures/fight.mp4'
        ]
        
        video_path = None
        for path in test_video_paths:
            if os.path.exists(path):
                video_path = path
                break
        
        if not video_path:
            pytest.skip("No fight video sample found")
        
        # Extract frame
        image_data = self.extract_frame(video_path)
        if not image_data:
            pytest.skip("Failed to extract frame from video")
        
        # Analyze
        result = analyze_image(
            image_data=image_data,
            ml_score=80,
            ml_factors={'poses': 2, 'aggressive_poses': 2},
            camera_id='test_camera'
        )
        
        # Verify score range (75-95)
        assert 75 <= result['aiScore'] <= 95, \
            f"Fight video score {result['aiScore']} not in expected range 75-95"
        
        # Verify scene type
        assert result['sceneType'] == 'real_fight', \
            f"Fight video classified as '{result['sceneType']}' instead of 'real_fight'"
        
        # Verify NOT prank/drama
        assert result['sceneType'] not in ['prank', 'drama'], \
            "Fight video should not be classified as prank/drama"
        
        # Verify score is NOT 40 (hardcoded fallback)
        assert result['aiScore'] != 40, \
            "Score should not be hardcoded 40"
        
        print(f"✓ Fight video: score={result['aiScore']}, scene={result['sceneType']}")
    
    @pytest.mark.skipif(not os.path.exists('ai-intelligence-layer/aiRouter_enhanced.py'),
                       reason="AI router not available")
    def test_boxing_video_classification(self, ai_router_available):
        """
        Test boxing videos → expect scores 20-35 (capped), scene type organized_sport.
        Requirements: 2.4, 2.5, 2.6
        """
        if not ai_router_available:
            pytest.skip("AI router not available")
        
        from aiRouter_enhanced import analyze_image
        
        # Look for boxing video samples
        test_video_paths = [
            'storage/test_videos/boxing_sample.mp4',
            'storage/recordings/boxing_test.mp4',
            'tests/fixtures/boxing.mp4'
        ]
        
        video_path = None
        for path in test_video_paths:
            if os.path.exists(path):
                video_path = path
                break
        
        if not video_path:
            pytest.skip("No boxing video sample found")
        
        # Extract frame
        image_data = self.extract_frame(video_path)
        if not image_data:
            pytest.skip("Failed to extract frame from video")
        
        # Analyze
        result = analyze_image(
            image_data=image_data,
            ml_score=50,
            ml_factors={'poses': 2, 'aggressive_poses': 1},
            camera_id='test_camera'
        )
        
        # Verify score range (20-35, capped)
        assert 20 <= result['aiScore'] <= 35, \
            f"Boxing video score {result['aiScore']} not in expected range 20-35"
        
        # Verify scene type
        assert result['sceneType'] == 'organized_sport', \
            f"Boxing video classified as '{result['sceneType']}' instead of 'organized_sport'"
        
        # Verify NOT prank/drama
        assert result['sceneType'] not in ['prank', 'drama'], \
            "Boxing video should not be classified as prank/drama"
        
        print(f"✓ Boxing video: score={result['aiScore']}, scene={result['sceneType']}")
    
    @pytest.mark.skipif(not os.path.exists('ai-intelligence-layer/aiRouter_enhanced.py'),
                       reason="AI router not available")
    def test_normal_video_classification(self, ai_router_available):
        """
        Test normal videos → expect scores 10-25, scene type normal.
        Requirements: 1.4, 2.7
        """
        if not ai_router_available:
            pytest.skip("AI router not available")
        
        from aiRouter_enhanced import analyze_image
        
        # Look for normal video samples
        test_video_paths = [
            'storage/test_videos/normal_sample.mp4',
            'storage/recordings/normal_test.mp4',
            'tests/fixtures/normal.mp4'
        ]
        
        video_path = None
        for path in test_video_paths:
            if os.path.exists(path):
                video_path = path
                break
        
        if not video_path:
            pytest.skip("No normal video sample found")
        
        # Extract frame
        image_data = self.extract_frame(video_path)
        if not image_data:
            pytest.skip("Failed to extract frame from video")
        
        # Analyze
        result = analyze_image(
            image_data=image_data,
            ml_score=15,
            ml_factors={'poses': 1, 'aggressive_poses': 0},
            camera_id='test_camera'
        )
        
        # Verify score range (10-25)
        assert 10 <= result['aiScore'] <= 25, \
            f"Normal video score {result['aiScore']} not in expected range 10-25"
        
        # Verify scene type
        assert result['sceneType'] == 'normal', \
            f"Normal video classified as '{result['sceneType']}' instead of 'normal'"
        
        # Verify NOT prank/drama
        assert result['sceneType'] not in ['prank', 'drama'], \
            "Normal video should not be classified as prank/drama"
        
        print(f"✓ Normal video: score={result['aiScore']}, scene={result['sceneType']}")
    
    @pytest.mark.skipif(not os.path.exists('ai-intelligence-layer/aiRouter_enhanced.py'),
                       reason="AI router not available")
    def test_suspicious_video_classification(self, ai_router_available):
        """
        Test suspicious behavior → expect scores 60-75, scene type suspicious.
        Requirements: 2.7
        """
        if not ai_router_available:
            pytest.skip("AI router not available")
        
        from aiRouter_enhanced import analyze_image
        
        # Look for suspicious video samples
        test_video_paths = [
            'storage/test_videos/suspicious_sample.mp4',
            'storage/recordings/suspicious_test.mp4',
            'tests/fixtures/suspicious.mp4'
        ]
        
        video_path = None
        for path in test_video_paths:
            if os.path.exists(path):
                video_path = path
                break
        
        if not video_path:
            pytest.skip("No suspicious video sample found")
        
        # Extract frame
        image_data = self.extract_frame(video_path)
        if not image_data:
            pytest.skip("Failed to extract frame from video")
        
        # Analyze
        result = analyze_image(
            image_data=image_data,
            ml_score=65,
            ml_factors={'poses': 3, 'crowd_behavior': 'surrounding'},
            camera_id='test_camera'
        )
        
        # Verify score range (60-75)
        assert 60 <= result['aiScore'] <= 75, \
            f"Suspicious video score {result['aiScore']} not in expected range 60-75"
        
        # Verify scene type
        assert result['sceneType'] == 'suspicious', \
            f"Suspicious video classified as '{result['sceneType']}' instead of 'suspicious'"
        
        # Verify NOT prank/drama
        assert result['sceneType'] not in ['prank', 'drama'], \
            "Suspicious video should not be classified as prank/drama"
        
        print(f"✓ Suspicious video: score={result['aiScore']}, scene={result['sceneType']}")
    
    @pytest.mark.skipif(not os.path.exists('ai-intelligence-layer/aiRouter_enhanced.py'),
                       reason="AI router not available")
    def test_no_hardcoded_40_scores(self, ai_router_available):
        """
        Test that scores do NOT cluster around 40 (hardcoded fallback).
        Requirements: 1.2, 1.5
        """
        if not ai_router_available:
            pytest.skip("AI router not available")
        
        from aiRouter_enhanced import analyze_image
        
        # Test multiple scenarios
        test_cases = [
            {'ml_score': 20, 'expected_range': (10, 30)},
            {'ml_score': 50, 'expected_range': (30, 70)},
            {'ml_score': 80, 'expected_range': (60, 95)},
        ]
        
        # Create test image
        img = Image.new('RGB', (640, 480), color='blue')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        image_data = f"data:image/jpeg;base64,{image_data}"
        
        scores = []
        for test_case in test_cases:
            result = analyze_image(
                image_data=image_data,
                ml_score=test_case['ml_score'],
                ml_factors={},
                camera_id='test_camera'
            )
            
            scores.append(result['aiScore'])
            
            # Verify score is NOT exactly 40
            assert result['aiScore'] != 40, \
                f"Score should not be hardcoded 40 (got {result['aiScore']})"
        
        # Verify scores have variety (not all clustering around 40)
        score_variance = np.var(scores)
        assert score_variance > 100, \
            f"Scores show low variance ({score_variance:.1f}), may be clustering around default value"
        
        print(f"✓ No hardcoded 40 scores: {scores} (variance: {score_variance:.1f})")
    
    @pytest.mark.skipif(not os.path.exists('ai-intelligence-layer/aiRouter_enhanced.py'),
                       reason="AI router not available")
    def test_no_prank_drama_classifications(self, ai_router_available):
        """
        Test that prank/drama classifications never appear.
        Requirements: 2.1, 2.2, 2.8
        """
        if not ai_router_available:
            pytest.skip("AI router not available")
        
        from aiRouter_enhanced import analyze_image
        
        # Test multiple scenarios
        test_cases = [
            {'ml_score': 30, 'description': 'low threat'},
            {'ml_score': 60, 'description': 'medium threat'},
            {'ml_score': 85, 'description': 'high threat'},
        ]
        
        # Create test image
        img = Image.new('RGB', (640, 480), color='red')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        image_data = f"data:image/jpeg;base64,{image_data}"
        
        for test_case in test_cases:
            result = analyze_image(
                image_data=image_data,
                ml_score=test_case['ml_score'],
                ml_factors={},
                camera_id='test_camera'
            )
            
            # Verify scene type is NOT prank or drama
            assert result['sceneType'] not in ['prank', 'drama'], \
                f"Scene type should not be prank/drama (got '{result['sceneType']}' for {test_case['description']})"
            
            # Verify scene type is one of the allowed values
            allowed_types = ['real_fight', 'organized_sport', 'normal', 'suspicious']
            assert result['sceneType'] in allowed_types, \
                f"Scene type '{result['sceneType']}' not in allowed types: {allowed_types}"
        
        print("✓ No prank/drama classifications found")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
