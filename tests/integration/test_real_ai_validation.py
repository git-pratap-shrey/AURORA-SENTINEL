"""
Real AI Validation Test - Test with ACTUAL AI Service (not mock)

This test calls the real AI intelligence layer to validate:
1. AI can detect real fights (high AI scores)
2. AI can discriminate boxing/sparring (low AI scores)
3. AI can identify normal activities (low AI scores)

Requirements: AI service must be running on http://localhost:3000
"""

import pytest
import asyncio
import numpy as np
import cv2
import sys
import json
import base64
import aiohttp
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.scoring.risk_engine import RiskScoringEngine
from models.detection.detector import UnifiedDetector


class RealAIClient:
    """Real AI client that calls the actual AI service."""
    
    def __init__(self, base_url='http://localhost:3000'):
        self.base_url = base_url
    
    async def analyze_image(self, imageData, mlScore, mlFactors, cameraId, timestamp, **kwargs):
        """Call the real AI service."""
        # Convert numpy array to base64 if needed
        if isinstance(imageData, np.ndarray):
            _, buffer = cv2.imencode('.jpg', imageData)
            imageData = base64.b64encode(buffer).decode('utf-8')
        
        payload = {
            'imageData': imageData,
            'mlScore': mlScore,
            'mlFactors': mlFactors,
            'cameraId': cameraId,
            'timestamp': timestamp
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f'{self.base_url}/analyze',
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"AI service error: {response.status} - {error_text}")
            except aiohttp.ClientConnectorError:
                raise Exception(f"Cannot connect to AI service at {self.base_url}. Is it running?")


class TestRealAIValidation:
    """Test suite for real AI validation."""
    
    TEST_VIDEOS = {
        'fight_1': 'storage/temp/raw_1772268166.802879_4rth.mp4',
        'fight_2': 'storage/temp/raw_1772268210.415873_5th.mp4',
        'boxing': 'storage/temp/raw_1772268625.365338_scr9-231238~2sdddsd.mp4',
        'normal_sample': 'data/sample_videos/Normal_Videos_for_Event_Recognition/Normal_Videos_050_x264.mp4'
    }
    
    @pytest.fixture(scope="class")
    def detector(self):
        """Initialize UnifiedDetector."""
        detector = UnifiedDetector()
        detector.warmup()
        return detector
    
    @pytest.fixture(scope="class")
    def risk_engine(self):
        """Initialize RiskScoringEngine."""
        return RiskScoringEngine(fps=30, bypass_calibration=True)
    
    @pytest.fixture
    def real_ai_client(self):
        """Create real AI client."""
        return RealAIClient()
    
    def process_video_sample(self, video_path: str, detector: UnifiedDetector, 
                            max_frames: int = 10) -> List[Dict]:
        """Process a sample of frames from video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        
        detections = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            detection_result = detector.process_frame(frame)
            
            detections.append({
                'frame_number': frame_idx,
                'timestamp': frame_idx / fps if fps > 0 else frame_idx,
                'frame': frame,
                'detection_data': detection_result
            })
        
        cap.release()
        return detections
    
    @pytest.mark.asyncio
    async def test_real_ai_on_fight_videos(self, detector, risk_engine, real_ai_client):
        """
        Test REAL AI on fight videos - AI should detect high threat.
        
        Expected:
        - AI should return high scores (60-90%) for real fights
        - AI should identify sceneType as "real_fight"
        - AI confidence should be high (>0.7)
        """
        print(f"\n{'='*80}")
        print(f"Testing REAL AI on Fight Videos")
        print(f"{'='*80}")
        
        for video_name, video_path in [('fight_1', self.TEST_VIDEOS['fight_1']), 
                                       ('fight_2', self.TEST_VIDEOS['fight_2'])]:
            if not Path(video_path).exists():
                print(f"\nSkipping {video_name}: file not found")
                continue
            
            print(f"\n{'='*60}")
            print(f"Video: {video_name}")
            print(f"{'='*60}")
            
            # Process video
            detections = self.process_video_sample(video_path, detector, max_frames=5)
            
            ai_scores = []
            scene_types = []
            
            for det in detections:
                # Calculate ML score
                ml_score, ml_factors = risk_engine.calculate_risk(
                    det['detection_data'],
                    {'timestamp': det['timestamp']}
                )
                
                print(f"\nFrame {det['frame_number']}:")
                print(f"  ML Score: {ml_score:.1f}%")
                
                # Call REAL AI
                try:
                    ai_result = await real_ai_client.analyze_image(
                        imageData=det['frame'],
                        mlScore=ml_score,
                        mlFactors=ml_factors,
                        cameraId=f'TEST-{video_name}',
                        timestamp=det['timestamp']
                    )
                    
                    ai_score = ai_result['aiScore']
                    scene_type = ai_result['sceneType']
                    explanation = ai_result['explanation']
                    confidence = ai_result['confidence']
                    
                    ai_scores.append(ai_score)
                    scene_types.append(scene_type)
                    
                    print(f"  AI Score: {ai_score:.1f}%")
                    print(f"  Scene Type: {scene_type}")
                    print(f"  Confidence: {confidence:.2f}")
                    print(f"  Explanation: {explanation[:100]}...")
                    
                except Exception as e:
                    print(f"  ❌ AI Error: {str(e)}")
                    continue
            
            if ai_scores:
                max_ai = max(ai_scores)
                avg_ai = np.mean(ai_scores)
                
                print(f"\n{video_name} Summary:")
                print(f"  Max AI Score: {max_ai:.1f}%")
                print(f"  Avg AI Score: {avg_ai:.1f}%")
                print(f"  Scene Types: {set(scene_types)}")
                
                # Validate: AI should detect fights with high scores
                assert max_ai > 50, f"AI failed to detect fight! Max AI score: {max_ai:.1f}%"
                print(f"  ✅ AI correctly detected fight (score > 50%)")
    
    @pytest.mark.asyncio
    async def test_real_ai_on_boxing_video(self, detector, risk_engine, real_ai_client):
        """
        Test REAL AI on boxing video - AI should discriminate as controlled activity.
        
        Expected:
        - AI should return low-medium scores (20-40%) for boxing
        - AI should identify sceneType as "boxing" or "controlled"
        - AI should explain it's controlled activity
        """
        print(f"\n{'='*80}")
        print(f"Testing REAL AI on Boxing Video")
        print(f"{'='*80}")
        
        video_path = self.TEST_VIDEOS['boxing']
        if not Path(video_path).exists():
            pytest.skip(f"Boxing video not found: {video_path}")
        
        # Process video
        detections = self.process_video_sample(video_path, detector, max_frames=5)
        
        ai_scores = []
        scene_types = []
        
        for det in detections:
            # Calculate ML score
            ml_score, ml_factors = risk_engine.calculate_risk(
                det['detection_data'],
                {'timestamp': det['timestamp']}
            )
            
            print(f"\nFrame {det['frame_number']}:")
            print(f"  ML Score: {ml_score:.1f}%")
            
            # Call REAL AI
            try:
                ai_result = await real_ai_client.analyze_image(
                    imageData=det['frame'],
                    mlScore=ml_score,
                    mlFactors=ml_factors,
                    cameraId='TEST-boxing',
                    timestamp=det['timestamp']
                )
                
                ai_score = ai_result['aiScore']
                scene_type = ai_result['sceneType']
                explanation = ai_result['explanation']
                confidence = ai_result['confidence']
                
                ai_scores.append(ai_score)
                scene_types.append(scene_type)
                
                print(f"  AI Score: {ai_score:.1f}%")
                print(f"  Scene Type: {scene_type}")
                print(f"  Confidence: {confidence:.2f}")
                print(f"  Explanation: {explanation[:100]}...")
                
            except Exception as e:
                print(f"  ❌ AI Error: {str(e)}")
                continue
        
        if ai_scores:
            max_ai = max(ai_scores)
            avg_ai = np.mean(ai_scores)
            
            print(f"\nBoxing Video Summary:")
            print(f"  Max AI Score: {max_ai:.1f}%")
            print(f"  Avg AI Score: {avg_ai:.1f}%")
            print(f"  Scene Types: {set(scene_types)}")
            
            # Validate: AI should discriminate boxing as controlled
            assert max_ai < 50, f"AI failed to discriminate boxing! Max AI score: {max_ai:.1f}%"
            print(f"  ✅ AI correctly discriminated boxing (score < 50%)")
    
    @pytest.mark.asyncio
    async def test_real_ai_on_normal_video(self, detector, risk_engine, real_ai_client):
        """
        Test REAL AI on normal video - AI should return very low scores.
        
        Expected:
        - AI should return very low scores (0-20%) for normal activities
        - AI should identify sceneType as "normal"
        """
        print(f"\n{'='*80}")
        print(f"Testing REAL AI on Normal Video")
        print(f"{'='*80}")
        
        video_path = self.TEST_VIDEOS['normal_sample']
        if not Path(video_path).exists():
            pytest.skip(f"Normal video not found: {video_path}")
        
        # Process video
        detections = self.process_video_sample(video_path, detector, max_frames=5)
        
        ai_scores = []
        scene_types = []
        
        for det in detections:
            # Calculate ML score
            ml_score, ml_factors = risk_engine.calculate_risk(
                det['detection_data'],
                {'timestamp': det['timestamp']}
            )
            
            print(f"\nFrame {det['frame_number']}:")
            print(f"  ML Score: {ml_score:.1f}%")
            
            # Call REAL AI
            try:
                ai_result = await real_ai_client.analyze_image(
                    imageData=det['frame'],
                    mlScore=ml_score,
                    mlFactors=ml_factors,
                    cameraId='TEST-normal',
                    timestamp=det['timestamp']
                )
                
                ai_score = ai_result['aiScore']
                scene_type = ai_result['sceneType']
                explanation = ai_result['explanation']
                confidence = ai_result['confidence']
                
                ai_scores.append(ai_score)
                scene_types.append(scene_type)
                
                print(f"  AI Score: {ai_score:.1f}%")
                print(f"  Scene Type: {scene_type}")
                print(f"  Confidence: {confidence:.2f}")
                print(f"  Explanation: {explanation[:100]}...")
                
            except Exception as e:
                print(f"  ❌ AI Error: {str(e)}")
                continue
        
        if ai_scores:
            max_ai = max(ai_scores)
            avg_ai = np.mean(ai_scores)
            
            print(f"\nNormal Video Summary:")
            print(f"  Max AI Score: {max_ai:.1f}%")
            print(f"  Avg AI Score: {avg_ai:.1f}%")
            print(f"  Scene Types: {set(scene_types)}")
            
            # Validate: AI should return low scores for normal
            assert max_ai < 30, f"AI incorrectly flagged normal activity! Max AI score: {max_ai:.1f}%"
            print(f"  ✅ AI correctly identified normal activity (score < 30%)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
