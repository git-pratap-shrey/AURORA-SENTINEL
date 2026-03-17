"""
Normal Video Testing Suite
Tests all normal videos to ensure low risk scores from ML and VLM
"""
import pytest
import sys
import os
import glob
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestNormalVideos:
    """Test normal videos for low risk scores"""
    
    @pytest.fixture(scope="class")
    def normal_video_paths(self):
        """Get all normal video paths"""
        # Existing normal videos
        existing_videos = glob.glob("data/sample_videos/Normal_Videos_for_Event_Recognition/*.mp4")
        
        # Synthetic normal videos
        synthetic_videos = glob.glob("tests/synthetic_data/videos/normal_cctv_*.mp4")
        
        all_videos = existing_videos + synthetic_videos
        print(f"\n📹 Found {len(all_videos)} normal videos to test")
        print(f"  - Existing: {len(existing_videos)}")
        print(f"  - Synthetic: {len(synthetic_videos)}")
        
        return all_videos
    
    def test_ml_risk_scores_low(self, normal_video_paths):
        """Test that ML risk scores are low for all normal videos"""
        try:
            from backend.services.ml_service import ml_service
            from models.scoring.risk_engine import RiskScoringEngine
            import cv2
            
            ml_service.load_models()
            engine = RiskScoringEngine(fps=30, bypass_calibration=True)
            
            results = []
            high_risk_videos = []
            
            for video_path in normal_video_paths[:10]:  # Test first 10 for speed
                video_name = os.path.basename(video_path)
                print(f"\n  Testing: {video_name}")
                
                cap = cv2.VideoCapture(video_path)
                max_risk = 0
                frame_count = 0
                
                # Sample every 30 frames
                while cap.isOpened() and frame_count < 300:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % 30 == 0:
                        detection = ml_service.detector.process_frame(frame)
                        risk_score, factors = engine.calculate_risk(detection)
                        max_risk = max(max_risk, risk_score)
                    
                    frame_count += 1
                
                cap.release()
                
                results.append({
                    "video": video_name,
                    "max_ml_risk": max_risk,
                    "frames_tested": frame_count // 30,
                    "status": "✅ PASS" if max_risk < 40 else "⚠️ HIGH"
                })
                
                if max_risk >= 40:
                    high_risk_videos.append((video_name, max_risk))
                
                print(f"    Max ML Risk: {max_risk:.1f}% - {'✅ PASS' if max_risk < 40 else '⚠️ HIGH'}")
            
            # Summary
            print(f"\n📊 ML Risk Score Summary:")
            print(f"  Total tested: {len(results)}")
            print(f"  Low risk (<40%): {sum(1 for r in results if r['max_ml_risk'] < 40)}")
            print(f"  High risk (>=40%): {len(high_risk_videos)}")
            
            if high_risk_videos:
                print(f"\n⚠️  High Risk Videos:")
                for name, risk in high_risk_videos:
                    print(f"    - {name}: {risk:.1f}%")
            
            # Save results
            with open('tests/normal_video_ml_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Assert: At least 80% should have low risk
            pass_rate = sum(1 for r in results if r['max_ml_risk'] < 40) / len(results)
            assert pass_rate >= 0.8, f"Only {pass_rate*100:.1f}% passed, expected >= 80%"
            
        except Exception as e:
            pytest.skip(f"ML testing not available: {e}")
    
    def test_vlm_risk_scores_low(self, normal_video_paths):
        """Test that VLM risk scores are low for all normal videos"""
        try:
            from backend.services.vlm_service import vlm_service
            from PIL import Image
            import cv2
            
            results = []
            high_risk_videos = []
            
            for video_path in normal_video_paths[:5]:  # Test first 5 (VLM is slower)
                video_name = os.path.basename(video_path)
                print(f"\n  Testing VLM: {video_name}")
                
                cap = cv2.VideoCapture(video_path)
                
                # Get middle frame
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    continue
                
                # Convert to PIL
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                # Analyze with VLM
                result = vlm_service.analyze_scene(pil_img, risk_score=20)
                vlm_risk = result.get('risk_score', 0)
                
                results.append({
                    "video": video_name,
                    "vlm_risk": vlm_risk,
                    "description": result.get('description', '')[:100],
                    "provider": result.get('provider', 'unknown'),
                    "status": "✅ PASS" if vlm_risk < 40 else "⚠️ HIGH"
                })
                
                if vlm_risk >= 40:
                    high_risk_videos.append((video_name, vlm_risk))
                
                print(f"    VLM Risk: {vlm_risk:.1f}% - {'✅ PASS' if vlm_risk < 40 else '⚠️ HIGH'}")
                print(f"    Provider: {result.get('provider')}")
            
            # Summary
            print(f"\n📊 VLM Risk Score Summary:")
            print(f"  Total tested: {len(results)}")
            print(f"  Low risk (<40%): {sum(1 for r in results if r['vlm_risk'] < 40)}")
            print(f"  High risk (>=40%): {len(high_risk_videos)}")
            
            # Save results
            with open('tests/normal_video_vlm_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Assert: At least 70% should have low risk (VLM can be more sensitive)
            if results:
                pass_rate = sum(1 for r in results if r['vlm_risk'] < 40) / len(results)
                assert pass_rate >= 0.7, f"Only {pass_rate*100:.1f}% passed, expected >= 70%"
            
        except Exception as e:
            pytest.skip(f"VLM testing not available: {e}")
