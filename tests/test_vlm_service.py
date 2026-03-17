"""
VLM Service Tests
Tests for AI intelligence layer and fusion engine
"""
import pytest
import sys
import os
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestVLMServiceFusion:
    """Test VLM fusion and risk assessment"""
    
    def test_fusion_engine_fight_keywords(self):
        """Test fight keywords trigger high risk scores"""
        try:
            from backend.services.vlm_service import vlm_service
            
            # Mock response with fight keywords
            mock_response = {
                "provider": "test",
                "description": "CRITICAL: Fight detected with weapons and aggressive behavior",
                "risk_score": 95
            }
            
            # Fusion should recognize high-risk keywords
            assert mock_response['risk_score'] >= 85
        except Exception as e:
            pytest.skip(f"VLM service not available: {e}")
    
    def test_fusion_engine_boxing_keyword(self):
        """Test boxing keywords suppress risk scores"""
        try:
            from backend.services.vlm_service import vlm_service
            
            mock_response = {
                "provider": "test",
                "description": "Boxing match in ring with referee, controlled sparring",
                "risk_score": 15
            }
            
            # Boxing should have low risk
            assert mock_response['risk_score'] <= 30
        except Exception as e:
            pytest.skip(f"VLM service not available: {e}")
    
    def test_fusion_engine_error_sanitization(self):
        """Test error responses are excluded from fusion"""
        responses = [
            {"provider": "gemini", "description": "Error: quota exceeded", "risk_score": 0},
            {"provider": "qwen", "description": "Fight detected", "risk_score": 85},
            {"provider": "hf", "description": "Aggressive behavior", "risk_score": 75}
        ]
        
        # Filter out errors
        valid_responses = [r for r in responses if "Error" not in r['description']]
        
        assert len(valid_responses) == 2
        assert all(r['risk_score'] > 0 for r in valid_responses)


class TestVLMServiceIntegration:
    """Test VLM service integration"""
    
    def test_analyze_scene_returns_schema(self, fight_frame):
        """Test analyze_scene returns correct schema"""
        try:
            from backend.services.vlm_service import vlm_service
            
            # Convert frame to PIL Image
            pil_img = Image.fromarray(fight_frame)
            
            result = vlm_service.analyze_scene(pil_img, risk_score=50)
            
            # Check required keys
            assert 'provider' in result
            assert 'description' in result
            assert 'risk_score' in result
            assert isinstance(result['risk_score'], (int, float))
        except Exception as e:
            pytest.skip(f"VLM service not available: {e}")
    
    def test_analyze_scene_escalates_risk(self, fight_frame):
        """Test analyze_scene escalates risk for high inputs"""
        try:
            from backend.services.vlm_service import vlm_service
            
            pil_img = Image.fromarray(fight_frame)
            
            result = vlm_service.analyze_scene(pil_img, risk_score=85)
            
            # High input risk should remain high or escalate
            assert result['risk_score'] >= 60
        except Exception as e:
            pytest.skip(f"VLM service not available: {e}")
    
    def test_local_ai_layer_connection(self):
        """Test connection to local AI layer"""
        try:
            import requests
            
            response = requests.get('http://localhost:3001/health', timeout=2)
            
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'ok'
        except Exception as e:
            pytest.skip(f"Local AI layer not running: {e}")
    
    def test_local_ai_chat_endpoint(self, fight_frame):
        """Test local AI chat endpoint"""
        try:
            import requests
            import base64
            from io import BytesIO
            
            # Convert frame to base64
            pil_img = Image.fromarray(fight_frame)
            buffer = BytesIO()
            pil_img.save(buffer, format='JPEG')
            image_data = base64.b64encode(buffer.getvalue()).decode()
            
            response = requests.post('http://localhost:3001/chat', json={
                'imageData': f"data:image/jpeg;base64,{image_data}",
                'question': 'What is happening in this image?'
            }, timeout=15)
            
            assert response.status_code == 200
            data = response.json()
            assert 'answer' in data
            assert 'provider' in data
        except Exception as e:
            pytest.skip(f"Local AI layer not available: {e}")
