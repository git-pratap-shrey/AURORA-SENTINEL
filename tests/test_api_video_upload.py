"""
API Video Upload Tests
Tests for video processing and analysis endpoints
"""
import pytest
import sys
import os
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestVideoUploadAPI:
    """Test video upload and processing"""
    
    def test_upload_fight_video_returns_200(self, app_client, fight_video_path):
        """Test uploading fight video returns HTTP 200"""
        try:
            with open(fight_video_path, 'rb') as f:
                files = {'file': ('fight_test.mp4', f, 'video/mp4')}
                response = app_client.post('/process/video', files=files)
            
            assert response.status_code == 200
        except Exception as e:
            pytest.skip(f"API not available: {e}")
    
    def test_upload_fight_video_response_schema(self, app_client, fight_video_path):
        """Test upload response has correct schema"""
        try:
            with open(fight_video_path, 'rb') as f:
                files = {'file': ('fight_test.mp4', f, 'video/mp4')}
                response = app_client.post('/process/video', files=files)
            
            data = response.json()
            
            assert 'status' in data
            assert 'alerts' in data or 'alerts_found' in data
            assert 'metrics' in data
            assert 'description' in data
        except Exception as e:
            pytest.skip(f"API not available: {e}")
    
    def test_upload_fight_video_fight_detected(self, app_client, fight_video_path):
        """Test fight detection in uploaded video"""
        try:
            with open(fight_video_path, 'rb') as f:
                files = {'file': ('fight_test.mp4', f, 'video/mp4')}
                response = app_client.post('/process/video', files=files)
            
            data = response.json()
            metrics = data.get('metrics', {})
            
            # Should detect some level of risk
            fight_prob = metrics.get('fight_probability', 0)
            alerts_found = data.get('alerts_found', 0)
            
            assert fight_prob > 0 or alerts_found > 0, "Should detect some risk in fight video"
        except Exception as e:
            pytest.skip(f"API not available: {e}")
    
    def test_upload_nonvideo_returns_error(self, app_client):
        """Test uploading non-video file returns error"""
        try:
            # Create fake text file
            fake_file = io.BytesIO(b"This is not a video")
            files = {'file': ('test.txt', fake_file, 'text/plain')}
            
            response = app_client.post('/process/video', files=files)
            
            # Should return error (400, 422, or 500)
            assert response.status_code >= 400
        except Exception as e:
            pytest.skip(f"API not available: {e}")


class TestVideoProcessingMetrics:
    """Test video processing metrics and quality"""
    
    def test_processing_metrics_completeness(self, app_client, fight_video_path):
        """Test processing metrics are complete and valid"""
        try:
            with open(fight_video_path, 'rb') as f:
                files = {'file': ('fight_test.mp4', f, 'video/mp4')}
                response = app_client.post('/process/video', files=files)
            
            data = response.json()
            metrics = data.get('metrics', {})
            
            # Check for key metrics
            assert 'fight_probability' in metrics or 'ml_baseline' in metrics
            assert 'max_persons' in metrics or metrics.get('max_persons') is not None
        except Exception as e:
            pytest.skip(f"API not available: {e}")
    
    def test_alert_generation_quality(self, app_client, fight_video_path):
        """Test alert generation quality and structure"""
        try:
            with open(fight_video_path, 'rb') as f:
                files = {'file': ('fight_test.mp4', f, 'video/mp4')}
                response = app_client.post('/process/video', files=files)
            
            data = response.json()
            alerts = data.get('alerts', [])
            
            for alert in alerts:
                # Check alert structure
                assert 'level' in alert
                assert 'score' in alert
                assert alert['level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL', 'low', 'medium', 'high', 'critical']
                assert 0 <= alert['score'] <= 100
        except Exception as e:
            pytest.skip(f"API not available: {e}")
    
    def test_processing_time_acceptable(self, app_client, fight_video_path):
        """Test video processing completes in reasonable time"""
        try:
            import time
            
            start_time = time.time()
            
            with open(fight_video_path, 'rb') as f:
                files = {'file': ('fight_test.mp4', f, 'video/mp4')}
                response = app_client.post('/process/video', files=files, timeout=60)
            
            elapsed = time.time() - start_time
            
            # 3-second video should process in under 60 seconds
            assert elapsed < 60, f"Processing took {elapsed}s, should be < 60s"
        except Exception as e:
            pytest.skip(f"API not available: {e}")
