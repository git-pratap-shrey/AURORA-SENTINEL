"""
Integration Tests for Nemotron Verification in analyze_image

Tests that analyze_image properly integrates Nemotron verification
after Qwen2-VL analysis.

**Validates: Requirements 3.1, 3.11, 3.12, 6.3, 7.2, 7.5**
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import base64
from io import BytesIO

# Add project root and ai-intelligence-layer to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai-intelligence-layer"))

from aiRouter_enhanced import analyze_image


class TestNemotronIntegration:
    """Test suite for Nemotron integration in analyze_image."""
    
    @pytest.fixture
    def sample_image_data(self):
        """Create a sample base64-encoded image."""
        img = Image.new('RGB', (640, 480), color='red')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{image_data}"
    
    @pytest.fixture
    def mock_qwen_result(self):
        """Mock Qwen2-VL analysis result."""
        return {
            'aiScore': 85,
            'sceneType': 'real_fight',
            'explanation': 'Two people engaged in physical fighting with punches',
            'confidence': 0.8,
            'provider': 'qwen2vl'
        }
    
    @pytest.fixture
    def mock_nemotron_verification(self):
        """Mock Nemotron verification result."""
        return {
            'verification_score': 0.75,
            'verified': True,
            'category_scores': {
                'real_fight': 0.82,
                'organized_sport': 0.15,
                'normal': 0.10,
                'suspicious': 0.20
            },
            'nemotron_scene_type': 'real_fight',
            'nemotron_risk_score': 88,
            'agreement': True,
            'recommended_score': 86,  # Average of 85 and 88
            'confidence': 0.9,
            'timed_out': False,
            'latency_ms': 250
        }
    
    def test_nemotron_called_after_qwen_success(self, sample_image_data, mock_qwen_result, mock_nemotron_verification):
        """
        Test that Nemotron is called after successful Qwen2-VL analysis.
        
        **Validates: Requirements 3.1, 3.11**
        """
        with patch('aiRouter_enhanced.analyze_with_qwen2vl') as mock_qwen, \
             patch('aiRouter_enhanced.init_nemotron') as mock_init_nemotron:
            
            # Setup mocks
            mock_qwen.return_value = mock_qwen_result.copy()
            
            mock_nemotron = Mock()
            mock_nemotron.available = True
            mock_nemotron.verify_analysis.return_value = mock_nemotron_verification.copy()
            mock_init_nemotron.return_value = mock_nemotron
            
            # Call analyze_image
            result = analyze_image(
                image_data=sample_image_data,
                ml_score=70,
                ml_factors={'aggression': 0.8},
                camera_id='TEST-CAM'
            )
            
            # Verify Nemotron was initialized
            mock_init_nemotron.assert_called_once()
            
            # Verify Nemotron verification was called
            mock_nemotron.verify_analysis.assert_called_once()
            
            # Verify the call arguments
            call_args = mock_nemotron.verify_analysis.call_args
            assert call_args[1]['qwen_summary'] == mock_qwen_result['explanation']
            assert call_args[1]['qwen_scene_type'] == mock_qwen_result['sceneType']
            assert call_args[1]['qwen_risk_score'] == mock_qwen_result['aiScore']
            assert call_args[1]['timeout'] == 3.0
            
            # Verify Nemotron verification is included in result
            assert 'nemotron_verification' in result
            assert result['nemotron_verification'] == mock_nemotron_verification
            assert 'nemotron_latency_ms' in result
    
    def test_nemotron_adjusted_score_used(self, sample_image_data, mock_qwen_result, mock_nemotron_verification):
        """
        Test that Nemotron's recommended score is used in final calculation.
        
        **Validates: Requirements 3.11, 3.12**
        """
        with patch('aiRouter_enhanced.analyze_with_qwen2vl') as mock_qwen, \
             patch('aiRouter_enhanced.init_nemotron') as mock_init_nemotron:
            
            # Setup mocks
            mock_qwen.return_value = mock_qwen_result.copy()
            
            mock_nemotron = Mock()
            mock_nemotron.available = True
            mock_nemotron.verify_analysis.return_value = mock_nemotron_verification.copy()
            mock_init_nemotron.return_value = mock_nemotron
            
            # Call analyze_image
            ml_score = 70
            result = analyze_image(
                image_data=sample_image_data,
                ml_score=ml_score,
                ml_factors={'aggression': 0.8},
                camera_id='TEST-CAM'
            )
            
            # Verify Nemotron's recommended score was used
            # ai_score_raw should be the Nemotron-adjusted score (86)
            assert result['ai_score_raw'] == mock_nemotron_verification['recommended_score']
            
            # Verify weighted calculation: 0.3 * ML + 0.7 * AI
            expected_final = int(0.3 * ml_score + 0.7 * mock_nemotron_verification['recommended_score'])
            assert result['aiScore'] == expected_final
    
    def test_nemotron_unavailable_fallback(self, sample_image_data, mock_qwen_result):
        """
        Test fallback when Nemotron is unavailable.
        
        **Validates: Requirements 3.11, 6.3**
        """
        with patch('aiRouter_enhanced.analyze_with_qwen2vl') as mock_qwen, \
             patch('aiRouter_enhanced.init_nemotron') as mock_init_nemotron:
            
            # Setup mocks - Nemotron unavailable
            mock_qwen.return_value = mock_qwen_result.copy()
            mock_init_nemotron.return_value = None  # Nemotron not available
            
            # Call analyze_image
            ml_score = 70
            result = analyze_image(
                image_data=sample_image_data,
                ml_score=ml_score,
                ml_factors={'aggression': 0.8},
                camera_id='TEST-CAM'
            )
            
            # Verify Nemotron verification is NOT in result
            assert 'nemotron_verification' not in result
            
            # Verify Qwen score is used directly
            # ai_score_raw should be the original Qwen score
            assert result['ai_score_raw'] == mock_qwen_result['aiScore']
            
            # Verify weighted calculation uses Qwen score
            expected_final = int(0.3 * ml_score + 0.7 * mock_qwen_result['aiScore'])
            assert result['aiScore'] == expected_final
    
    def test_nemotron_timeout_fallback(self, sample_image_data, mock_qwen_result):
        """
        Test fallback when Nemotron times out.
        
        **Validates: Requirements 6.3, 7.2**
        """
        with patch('aiRouter_enhanced.analyze_with_qwen2vl') as mock_qwen, \
             patch('aiRouter_enhanced.init_nemotron') as mock_init_nemotron:
            
            # Setup mocks - Nemotron times out
            mock_qwen.return_value = mock_qwen_result.copy()
            
            mock_nemotron = Mock()
            mock_nemotron.available = True
            mock_nemotron.verify_analysis.return_value = {
                'verification_score': 0.0,
                'verified': False,
                'category_scores': {'real_fight': 0.0, 'organized_sport': 0.0, 'normal': 0.0, 'suspicious': 0.0},
                'nemotron_scene_type': 'real_fight',
                'nemotron_risk_score': 85,
                'agreement': True,
                'recommended_score': 85,
                'confidence': 0.6,
                'timed_out': True,  # Timeout occurred
                'latency_ms': 3100
            }
            mock_init_nemotron.return_value = mock_nemotron
            
            # Call analyze_image
            ml_score = 70
            result = analyze_image(
                image_data=sample_image_data,
                ml_score=ml_score,
                ml_factors={'aggression': 0.8},
                camera_id='TEST-CAM'
            )
            
            # Verify Nemotron verification is included but timed out
            assert 'nemotron_verification' in result
            assert result['nemotron_verification']['timed_out'] is True
            
            # Verify Qwen score is used (not Nemotron's)
            assert result['ai_score_raw'] == mock_qwen_result['aiScore']
    
    def test_nemotron_exception_handling(self, sample_image_data, mock_qwen_result):
        """
        Test exception handling when Nemotron verification fails.
        
        **Validates: Requirements 6.3**
        """
        with patch('aiRouter_enhanced.analyze_with_qwen2vl') as mock_qwen, \
             patch('aiRouter_enhanced.init_nemotron') as mock_init_nemotron:
            
            # Setup mocks - Nemotron raises exception
            mock_qwen.return_value = mock_qwen_result.copy()
            
            mock_nemotron = Mock()
            mock_nemotron.available = True
            mock_nemotron.verify_analysis.side_effect = RuntimeError("Nemotron verification failed")
            mock_init_nemotron.return_value = mock_nemotron
            
            # Call analyze_image - should not raise exception
            ml_score = 70
            result = analyze_image(
                image_data=sample_image_data,
                ml_score=ml_score,
                ml_factors={'aggression': 0.8},
                camera_id='TEST-CAM'
            )
            
            # Verify analysis completes successfully
            assert result is not None
            assert 'aiScore' in result
            
            # Verify Nemotron verification is NOT in result
            assert 'nemotron_verification' not in result
            
            # Verify Qwen score is used
            assert result['ai_score_raw'] == mock_qwen_result['aiScore']
    
    def test_latency_tracking(self, sample_image_data, mock_qwen_result, mock_nemotron_verification):
        """
        Test that Nemotron latency is tracked.
        
        **Validates: Requirements 7.2, 7.5**
        """
        with patch('aiRouter_enhanced.analyze_with_qwen2vl') as mock_qwen, \
             patch('aiRouter_enhanced.init_nemotron') as mock_init_nemotron:
            
            # Setup mocks
            mock_qwen.return_value = mock_qwen_result.copy()
            
            mock_nemotron = Mock()
            mock_nemotron.available = True
            mock_nemotron.verify_analysis.return_value = mock_nemotron_verification.copy()
            mock_init_nemotron.return_value = mock_nemotron
            
            # Call analyze_image
            result = analyze_image(
                image_data=sample_image_data,
                ml_score=70,
                ml_factors={'aggression': 0.8},
                camera_id='TEST-CAM'
            )
            
            # Verify latency is tracked
            assert 'nemotron_latency_ms' in result
            assert isinstance(result['nemotron_latency_ms'], int)
            assert result['nemotron_latency_ms'] >= 0
    
    def test_low_ml_score_skips_analysis(self, sample_image_data):
        """
        Test that low ML scores skip AI analysis entirely.
        
        **Validates: Requirements 7.6**
        """
        with patch('aiRouter_enhanced.analyze_with_qwen2vl') as mock_qwen, \
             patch('aiRouter_enhanced.init_nemotron') as mock_init_nemotron:
            
            # Call analyze_image with low ML score
            result = analyze_image(
                image_data=sample_image_data,
                ml_score=15,  # Below threshold of 20
                ml_factors={},
                camera_id='TEST-CAM'
            )
            
            # Verify Qwen was NOT called
            mock_qwen.assert_not_called()
            
            # Verify Nemotron was NOT called
            mock_init_nemotron.assert_not_called()
            
            # Verify result indicates no analysis
            assert result['provider'] == 'none'
            assert result['sceneType'] == 'normal'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
