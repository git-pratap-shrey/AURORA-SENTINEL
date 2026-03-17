"""
Unit tests for error handling in AI scoring system
Tests Requirements 6.1, 6.2, 6.3, 6.4, 6.7, 6.8
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import base64
from io import BytesIO
from PIL import Image

# Add ai-intelligence-layer to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ai-intelligence-layer'))

from model_availability import ModelAvailabilityTracker


class TestQwenFailureOllamaFallback:
    """Test Qwen failure → Ollama fallback (Requirement 6.1)"""
    
    def setup_method(self):
        """Reset global state before each test"""
        # Reset module-level variables
        import aiRouter_enhanced
        aiRouter_enhanced._qwen2vl_analyzer = None
        aiRouter_enhanced._ollama_available = False
        aiRouter_enhanced._nemotron_provider = None
        aiRouter_enhanced._availability_tracker = ModelAvailabilityTracker()
    
    def create_test_image(self):
        """Create a test image and encode as base64"""
        img = Image.new('RGB', (640, 480), color='red')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{image_data}"
    
    @patch('aiRouter_enhanced.init_qwen2vl')
    @patch('aiRouter_enhanced.init_ollama')
    @patch('aiRouter_enhanced.analyze_with_ollama')
    def test_qwen_failure_triggers_ollama_fallback(self, mock_analyze_ollama, mock_init_ollama, mock_init_qwen):
        """Test that Qwen failure triggers Ollama fallback"""
        from aiRouter_enhanced import analyze_image
        
        # Qwen fails to initialize
        mock_init_qwen.return_value = None
        
        # Ollama is available
        mock_init_ollama.return_value = True
        mock_analyze_ollama.return_value = {
            'aiScore': 75,
            'sceneType': 'real_fight',
            'explanation': 'Fight detected by Ollama',
            'confidence': 0.8,
            'provider': 'ollama'
        }
        
        image_data = self.create_test_image()
        result = analyze_image(image_data, ml_score=70, ml_factors={}, camera_id='TEST')
        
        # Verify Ollama was called as fallback
        assert mock_analyze_ollama.called
        assert result['provider'] == 'ollama'
        assert result['aiScore'] is not None
        assert 'errors' in result
        assert 'qwen2vl' in result['errors']
    
    @patch('aiRouter_enhanced.analyze_with_qwen2vl')
    @patch('aiRouter_enhanced.init_ollama')
    @patch('aiRouter_enhanced.analyze_with_ollama')
    def test_qwen_analysis_failure_triggers_ollama(self, mock_analyze_ollama, mock_init_ollama, mock_analyze_qwen):
        """Test that Qwen analysis failure (not init) triggers Ollama"""
        from aiRouter_enhanced import analyze_image
        
        # Qwen analysis fails
        mock_analyze_qwen.return_value = None
        
        # Ollama succeeds
        mock_init_ollama.return_value = True
        mock_analyze_ollama.return_value = {
            'aiScore': 80,
            'sceneType': 'real_fight',
            'explanation': 'Fight detected by Ollama fallback',
            'confidence': 0.75,
            'provider': 'ollama'
        }
        
        image_data = self.create_test_image()
        result = analyze_image(image_data, ml_score=75, ml_factors={}, camera_id='TEST')
        
        # Verify fallback occurred
        assert result['provider'] == 'ollama'
        assert result['aiScore'] is not None
        assert 'errors' in result


class TestNemotronFailureSingleModel:
    """Test Nemotron failure → single-model analysis (Requirement 6.2)"""
    
    def setup_method(self):
        """Reset global state before each test"""
        import aiRouter_enhanced
        aiRouter_enhanced._qwen2vl_analyzer = None
        aiRouter_enhanced._ollama_available = False
        aiRouter_enhanced._nemotron_provider = None
        aiRouter_enhanced._availability_tracker = ModelAvailabilityTracker()
    
    def create_test_image(self):
        """Create a test image and encode as base64"""
        img = Image.new('RGB', (640, 480), color='red')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{image_data}"
    
    @patch('aiRouter_enhanced.analyze_with_qwen2vl')
    @patch('aiRouter_enhanced.init_nemotron')
    def test_nemotron_init_failure_continues_with_qwen(self, mock_init_nemotron, mock_analyze_qwen):
        """Test that Nemotron init failure continues with Qwen score"""
        from aiRouter_enhanced import analyze_image
        
        # Qwen succeeds
        mock_analyze_qwen.return_value = {
            'aiScore': 85,
            'sceneType': 'real_fight',
            'explanation': 'Fight detected by Qwen',
            'confidence': 0.8,
            'provider': 'qwen2vl'
        }
        
        # Nemotron fails to initialize
        mock_init_nemotron.return_value = None
        
        image_data = self.create_test_image()
        result = analyze_image(image_data, ml_score=80, ml_factors={}, camera_id='TEST')
        
        # Verify Qwen score is used without Nemotron verification
        assert result['provider'] == 'qwen2vl'
        assert result['aiScore'] is not None
        assert 'nemotron_verification' not in result or result.get('nemotron_verification') is None
        assert 'errors' in result
        assert 'nemotron' in result['errors']
    
    @patch('aiRouter_enhanced.analyze_with_qwen2vl')
    @patch('aiRouter_enhanced.init_nemotron')
    def test_nemotron_verification_failure_uses_qwen_score(self, mock_init_nemotron, mock_analyze_qwen):
        """Test that Nemotron verification failure uses Qwen score"""
        from aiRouter_enhanced import analyze_image
        
        # Qwen succeeds
        mock_analyze_qwen.return_value = {
            'aiScore': 75,
            'sceneType': 'real_fight',
            'explanation': 'Fight detected',
            'confidence': 0.8,
            'provider': 'qwen2vl'
        }
        
        # Nemotron initializes but verification fails
        mock_nemotron = Mock()
        mock_nemotron.available = True
        mock_nemotron.verify_analysis.side_effect = RuntimeError("Verification failed")
        mock_init_nemotron.return_value = mock_nemotron
        
        image_data = self.create_test_image()
        result = analyze_image(image_data, ml_score=70, ml_factors={}, camera_id='TEST')
        
        # Verify Qwen score is used
        assert result['provider'] == 'qwen2vl'
        assert result['aiScore'] is not None
        assert 'errors' in result
        assert 'nemotron' in result['errors']


class TestNemotronTimeout:
    """Test Nemotron timeout → use Qwen score (Requirement 6.3)"""
    
    def setup_method(self):
        """Reset global state before each test"""
        import aiRouter_enhanced
        aiRouter_enhanced._qwen2vl_analyzer = None
        aiRouter_enhanced._ollama_available = False
        aiRouter_enhanced._nemotron_provider = None
        aiRouter_enhanced._availability_tracker = ModelAvailabilityTracker()
    
    def create_test_image(self):
        """Create a test image and encode as base64"""
        img = Image.new('RGB', (640, 480), color='red')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{image_data}"
    
    @patch('aiRouter_enhanced.analyze_with_qwen2vl')
    @patch('aiRouter_enhanced.init_nemotron')
    def test_nemotron_timeout_uses_qwen_score(self, mock_init_nemotron, mock_analyze_qwen):
        """Test that Nemotron timeout uses Qwen score"""
        from aiRouter_enhanced import analyze_image
        
        # Qwen succeeds
        mock_analyze_qwen.return_value = {
            'aiScore': 82,
            'sceneType': 'real_fight',
            'explanation': 'Fight detected',
            'confidence': 0.8,
            'provider': 'qwen2vl'
        }
        
        # Nemotron times out
        mock_nemotron = Mock()
        mock_nemotron.available = True
        mock_nemotron.verify_analysis.return_value = {
            'verification_score': 0.0,
            'verified': False,
            'category_scores': {},
            'nemotron_scene_type': 'real_fight',
            'nemotron_risk_score': 82,
            'agreement': True,
            'recommended_score': 82,
            'confidence': 0.6,
            'timed_out': True,
            'latency_ms': 3100
        }
        mock_init_nemotron.return_value = mock_nemotron
        
        image_data = self.create_test_image()
        result = analyze_image(image_data, ml_score=75, ml_factors={}, camera_id='TEST')
        
        # Verify Qwen score is used when timeout occurs
        assert result['provider'] == 'qwen2vl'
        assert result['aiScore'] is not None
        assert 'nemotron_verification' in result
        assert result['nemotron_verification']['timed_out'] is True
        assert 'errors' in result
        assert 'nemotron' in result['errors']


class TestBothAIModelsFail:
    """Test both AI models fail → ML score used (Requirement 6.4)"""
    
    def setup_method(self):
        """Reset global state before each test"""
        import aiRouter_enhanced
        aiRouter_enhanced._qwen2vl_analyzer = None
        aiRouter_enhanced._ollama_available = False
        aiRouter_enhanced._nemotron_provider = None
        aiRouter_enhanced._availability_tracker = ModelAvailabilityTracker()
    
    def create_test_image(self):
        """Create a test image and encode as base64"""
        img = Image.new('RGB', (640, 480), color='red')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{image_data}"
    
    @patch('aiRouter_enhanced.analyze_with_qwen2vl')
    @patch('aiRouter_enhanced.analyze_with_ollama')
    def test_both_models_fail_uses_ml_score(self, mock_analyze_ollama, mock_analyze_qwen):
        """Test that both AI models failing uses ML score as final"""
        from aiRouter_enhanced import analyze_image
        
        # Both models fail
        mock_analyze_qwen.return_value = None
        mock_analyze_ollama.return_value = None
        
        image_data = self.create_test_image()
        ml_score = 78
        result = analyze_image(image_data, ml_score=ml_score, ml_factors={'aggression': 0.8}, camera_id='TEST')
        
        # Verify ML score is used as final
        assert result['provider'] == 'ml_fallback'
        assert result['aiScore'] == ml_score
        assert result['confidence'] == 0.3  # Low confidence for ML-only
        assert 'errors' in result
        assert 'qwen2vl' in result['errors']
        assert 'ollama' in result['errors']
    
    @patch('aiRouter_enhanced.analyze_with_qwen2vl')
    @patch('aiRouter_enhanced.analyze_with_ollama')
    def test_ml_score_with_weapon_detection(self, mock_analyze_ollama, mock_analyze_qwen):
        """Test ML score fallback with weapon detection"""
        from aiRouter_enhanced import analyze_image
        
        # Both models fail
        mock_analyze_qwen.return_value = None
        mock_analyze_ollama.return_value = None
        
        image_data = self.create_test_image()
        ml_score = 65
        ml_factors = {'weapon': True, 'aggression': 0.9}
        result = analyze_image(image_data, ml_score=ml_score, ml_factors=ml_factors, camera_id='TEST')
        
        # Verify ML score is adjusted for weapon
        assert result['provider'] == 'ml_fallback'
        assert result['aiScore'] >= 80  # Weapon detection boosts score
        assert result['sceneType'] == 'real_fight'
        assert 'weapon' in result['explanation'].lower()


class TestConsecutiveFailureTracking:
    """Test consecutive failure tracking (Requirement 6.7)"""
    
    def setup_method(self):
        """Reset global state before each test"""
        import aiRouter_enhanced
        aiRouter_enhanced._qwen2vl_analyzer = None
        aiRouter_enhanced._ollama_available = False
        aiRouter_enhanced._nemotron_provider = None
        aiRouter_enhanced._availability_tracker = ModelAvailabilityTracker()
    
    def create_test_image(self):
        """Create a test image and encode as base64"""
        img = Image.new('RGB', (640, 480), color='red')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{image_data}"
    
    @patch('aiRouter_enhanced.init_qwen2vl')
    @patch('aiRouter_enhanced.analyze_with_ollama')
    def test_three_consecutive_failures_marks_unavailable(self, mock_analyze_ollama, mock_init_qwen):
        """Test that 3 consecutive failures marks model unavailable"""
        from aiRouter_enhanced import analyze_image, _availability_tracker
        
        # Qwen fails to initialize 3 times
        mock_init_qwen.return_value = None
        
        # Ollama succeeds
        mock_analyze_ollama.return_value = {
            'aiScore': 70,
            'sceneType': 'real_fight',
            'explanation': 'Fight detected',
            'confidence': 0.75,
            'provider': 'ollama'
        }
        
        image_data = self.create_test_image()
        
        # First failure
        analyze_image(image_data, ml_score=70, ml_factors={}, camera_id='TEST')
        assert _availability_tracker.get_status('qwen2vl')['consecutive_failures'] == 1
        assert _availability_tracker.is_available('qwen2vl')
        
        # Second failure
        analyze_image(image_data, ml_score=70, ml_factors={}, camera_id='TEST')
        assert _availability_tracker.get_status('qwen2vl')['consecutive_failures'] == 2
        assert _availability_tracker.is_available('qwen2vl')
        
        # Third failure - should mark unavailable
        analyze_image(image_data, ml_score=70, ml_factors={}, camera_id='TEST')
        assert _availability_tracker.get_status('qwen2vl')['consecutive_failures'] == 3
        assert not _availability_tracker.is_available('qwen2vl')
    
    def test_success_resets_consecutive_failures(self):
        """Test that success resets consecutive failure count"""
        from aiRouter_enhanced import _availability_tracker
        
        # Directly test the tracker behavior
        # Two failures
        _availability_tracker.record_failure('qwen2vl')
        _availability_tracker.record_failure('qwen2vl')
        assert _availability_tracker.get_status('qwen2vl')['consecutive_failures'] == 2
        
        # Success resets
        _availability_tracker.record_success('qwen2vl')
        assert _availability_tracker.get_status('qwen2vl')['consecutive_failures'] == 0


class TestModelUnavailableSkipLogic:
    """Test model unavailable skip logic with 5 minute cooldown (Requirement 6.7)"""
    
    def setup_method(self):
        """Reset global state before each test"""
        import aiRouter_enhanced
        aiRouter_enhanced._qwen2vl_analyzer = None
        aiRouter_enhanced._ollama_available = False
        aiRouter_enhanced._nemotron_provider = None
        aiRouter_enhanced._availability_tracker = ModelAvailabilityTracker()
        # Reduce cooldown for testing
        aiRouter_enhanced._availability_tracker.COOLDOWN_SECONDS = 1
    
    def create_test_image(self):
        """Create a test image and encode as base64"""
        img = Image.new('RGB', (640, 480), color='red')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{image_data}"
    
    @patch('aiRouter_enhanced.init_qwen2vl')
    @patch('aiRouter_enhanced.analyze_with_ollama')
    def test_unavailable_model_skipped_during_cooldown(self, mock_analyze_ollama, mock_init_qwen):
        """Test that unavailable model is skipped during cooldown"""
        from aiRouter_enhanced import analyze_image, _availability_tracker
        import time
        
        # Qwen fails 3 times
        mock_init_qwen.return_value = None
        mock_analyze_ollama.return_value = {
            'aiScore': 70,
            'sceneType': 'real_fight',
            'explanation': 'Fight detected',
            'confidence': 0.75,
            'provider': 'ollama'
        }
        
        image_data = self.create_test_image()
        
        # Trigger 3 failures
        for _ in range(3):
            analyze_image(image_data, ml_score=70, ml_factors={}, camera_id='TEST')
        
        # Verify model is unavailable
        assert not _availability_tracker.is_available('qwen2vl')
        
        # Reset mock call count
        mock_init_qwen.reset_mock()
        
        # Try again immediately - should skip Qwen
        analyze_image(image_data, ml_score=70, ml_factors={}, camera_id='TEST')
        assert not mock_init_qwen.called  # Should not attempt to initialize
        
        # Wait for cooldown
        time.sleep(1.1)
        
        # Try again - should retry Qwen
        analyze_image(image_data, ml_score=70, ml_factors={}, camera_id='TEST')
        # Note: init_qwen2vl might be called now that cooldown expired


class TestAllErrorPathsReturnValidScores:
    """Test that all error paths return valid scores (Requirement 6.8)"""
    
    def setup_method(self):
        """Reset global state before each test"""
        import aiRouter_enhanced
        aiRouter_enhanced._qwen2vl_analyzer = None
        aiRouter_enhanced._ollama_available = False
        aiRouter_enhanced._nemotron_provider = None
        aiRouter_enhanced._availability_tracker = ModelAvailabilityTracker()
    
    def create_test_image(self):
        """Create a test image and encode as base64"""
        img = Image.new('RGB', (640, 480), color='red')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{image_data}"
    
    @patch('aiRouter_enhanced.analyze_with_qwen2vl')
    @patch('aiRouter_enhanced.analyze_with_ollama')
    def test_all_models_fail_returns_valid_score(self, mock_analyze_ollama, mock_analyze_qwen):
        """Test that all models failing still returns valid score"""
        from aiRouter_enhanced import analyze_image
        
        # All models fail
        mock_analyze_qwen.return_value = None
        mock_analyze_ollama.return_value = None
        
        image_data = self.create_test_image()
        result = analyze_image(image_data, ml_score=75, ml_factors={}, camera_id='TEST')
        
        # Verify valid score is returned
        assert result is not None
        assert 'aiScore' in result
        assert result['aiScore'] is not None
        assert isinstance(result['aiScore'], (int, float))
        assert 0 <= result['aiScore'] <= 100
        assert result['aiScore'] == 75  # Should equal ML score
    
    def test_invalid_image_returns_valid_score(self):
        """Test that invalid image still returns valid score"""
        from aiRouter_enhanced import analyze_image
        
        # Invalid base64 image
        result = analyze_image(
            image_data="invalid_base64_data",
            ml_score=60,
            ml_factors={},
            camera_id='TEST'
        )
        
        # Verify valid score is returned
        assert result is not None
        assert 'aiScore' in result
        assert result['aiScore'] is not None
        assert isinstance(result['aiScore'], (int, float))
        assert result['aiScore'] == 60  # Should equal ML score
    
    def test_none_ml_score_returns_valid_score(self):
        """Test that None ML score is handled gracefully"""
        from aiRouter_enhanced import analyze_image
        
        image_data = self.create_test_image()
        result = analyze_image(
            image_data=image_data,
            ml_score=None,
            ml_factors={},
            camera_id='TEST'
        )
        
        # Verify valid score is returned
        assert result is not None
        assert 'aiScore' in result
        assert result['aiScore'] is not None
        assert isinstance(result['aiScore'], (int, float))
        assert result['aiScore'] == 0  # Should default to 0
    
    @patch('aiRouter_enhanced.analyze_with_qwen2vl')
    @patch('aiRouter_enhanced.analyze_with_ollama')
    def test_qwen_returns_none_score_uses_ml_fallback(self, mock_analyze_ollama, mock_analyze_qwen):
        """Test that Qwen returning None score is treated as failure and uses fallback"""
        from aiRouter_enhanced import analyze_image
        
        # Qwen returns None (complete failure)
        mock_analyze_qwen.return_value = None
        
        # Ollama also fails
        mock_analyze_ollama.return_value = None
        
        image_data = self.create_test_image()
        result = analyze_image(image_data, ml_score=65, ml_factors={}, camera_id='TEST')
        
        # Verify ML score is used as fallback when both AI models fail
        assert result is not None
        assert result['aiScore'] is not None
        assert isinstance(result['aiScore'], (int, float))
        assert result['aiScore'] == 65  # Should use ML score
        assert result['provider'] == 'ml_fallback'


class TestErrorDetailsInResponse:
    """Test that error details are included in response (Requirement 6.5)"""
    
    def setup_method(self):
        """Reset global state before each test"""
        import aiRouter_enhanced
        aiRouter_enhanced._qwen2vl_analyzer = None
        aiRouter_enhanced._ollama_available = False
        aiRouter_enhanced._nemotron_provider = None
        aiRouter_enhanced._availability_tracker = ModelAvailabilityTracker()
    
    def create_test_image(self):
        """Create a test image and encode as base64"""
        img = Image.new('RGB', (640, 480), color='red')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{image_data}"
    
    @patch('aiRouter_enhanced.analyze_with_qwen2vl')
    @patch('aiRouter_enhanced.analyze_with_ollama')
    def test_error_details_included_when_models_fail(self, mock_analyze_ollama, mock_analyze_qwen):
        """Test that error details are included when models fail"""
        from aiRouter_enhanced import analyze_image
        
        # Both models fail
        mock_analyze_qwen.return_value = None
        mock_analyze_ollama.return_value = None
        
        image_data = self.create_test_image()
        result = analyze_image(image_data, ml_score=70, ml_factors={}, camera_id='TEST')
        
        # Verify error details are included
        assert 'errors' in result
        assert isinstance(result['errors'], dict)
        assert 'qwen2vl' in result['errors']
        assert 'ollama' in result['errors']
    
    @patch('aiRouter_enhanced.analyze_with_qwen2vl')
    @patch('aiRouter_enhanced.init_nemotron')
    def test_nemotron_error_details_included(self, mock_init_nemotron, mock_analyze_qwen):
        """Test that Nemotron error details are included"""
        from aiRouter_enhanced import analyze_image
        
        # Qwen succeeds
        mock_analyze_qwen.return_value = {
            'aiScore': 75,
            'sceneType': 'real_fight',
            'explanation': 'Fight detected',
            'confidence': 0.8,
            'provider': 'qwen2vl'
        }
        
        # Nemotron fails
        mock_init_nemotron.return_value = None
        
        image_data = self.create_test_image()
        result = analyze_image(image_data, ml_score=70, ml_factors={}, camera_id='TEST')
        
        # Verify Nemotron error is included
        assert 'errors' in result
        assert 'nemotron' in result['errors']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
