"""
Unit tests for Nemotron verification logic
Tests agreement logic, score recommendation, and timeout handling
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestNemotronAgreementLogic:
    """Test Nemotron agreement logic and score recommendation"""
    
    def test_verified_agreement_uses_average_score(self):
        """Test that verified agreement uses average of both scores with confidence 0.9"""
        from backend.services.vlm_service import NemotronProvider
        
        # Create mock provider
        provider = NemotronProvider()
        if not provider.available:
            pytest.skip("Nemotron model not available")
        
        # Mock the embedding methods
        with patch.object(provider, 'forward_images') as mock_images, \
             patch.object(provider, 'forward_queries') as mock_queries, \
             patch.object(provider, 'get_scores') as mock_scores:
            
            # Setup mocks
            mock_images.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_queries.return_value = np.array([[0.1, 0.2, 0.3]])
            
            # Verification score > 0.6 (verified)
            # Category scores: real_fight highest (agreement with Qwen)
            mock_scores.side_effect = [
                np.array([[0.75]]),  # Verification score
                np.array([[0.85, 0.2, 0.1, 0.15]])  # Category scores
            ]
            
            # Create test image
            test_image = Image.new('RGB', (100, 100))
            
            result = provider.verify_analysis(
                image=test_image,
                qwen_summary="Fight detected",
                qwen_scene_type="real_fight",
                qwen_risk_score=85
            )
            
            # Assertions
            assert result['verified'] is True
            assert result['agreement'] is True
            assert result['nemotron_scene_type'] == 'real_fight'
            
            # Should use average: (85 + nemotron_score) / 2
            # Nemotron score should be in range 80-95 for real_fight with score > 0.7
            expected_nemotron_score = int(np.mean([80, 95]))  # 87
            expected_recommended = int((85 + expected_nemotron_score) / 2)  # 86
            
            assert result['recommended_score'] == expected_recommended
            assert result['confidence'] == 0.9
            assert result['timed_out'] is False
    
    def test_mismatch_uses_higher_risk_score(self):
        """Test that verification mismatch uses higher risk score with confidence 0.5"""
        from backend.services.vlm_service import NemotronProvider
        
        provider = NemotronProvider()
        if not provider.available:
            pytest.skip("Nemotron model not available")
        
        with patch.object(provider, 'forward_images') as mock_images, \
             patch.object(provider, 'forward_queries') as mock_queries, \
             patch.object(provider, 'get_scores') as mock_scores:
            
            mock_images.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_queries.return_value = np.array([[0.1, 0.2, 0.3]])
            
            # Verification score < 0.6 (mismatch)
            mock_scores.side_effect = [
                np.array([[0.45]]),  # Verification score (mismatch)
                np.array([[0.85, 0.2, 0.1, 0.15]])  # Category scores
            ]
            
            test_image = Image.new('RGB', (100, 100))
            
            result = provider.verify_analysis(
                image=test_image,
                qwen_summary="Normal activity",
                qwen_scene_type="normal",
                qwen_risk_score=20
            )
            
            # Assertions
            assert result['verified'] is False
            assert result['nemotron_scene_type'] == 'real_fight'
            
            # Should use higher risk score (conservative)
            expected_nemotron_score = int(np.mean([80, 95]))  # 87
            expected_recommended = max(20, expected_nemotron_score)  # 87
            
            assert result['recommended_score'] == expected_recommended
            assert result['confidence'] == 0.5
            assert result['timed_out'] is False
    
    def test_disagreement_uses_higher_risk_score(self):
        """Test that scene type disagreement uses higher risk score"""
        from backend.services.vlm_service import NemotronProvider
        
        provider = NemotronProvider()
        if not provider.available:
            pytest.skip("Nemotron model not available")
        
        with patch.object(provider, 'forward_images') as mock_images, \
             patch.object(provider, 'forward_queries') as mock_queries, \
             patch.object(provider, 'get_scores') as mock_scores:
            
            mock_images.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_queries.return_value = np.array([[0.1, 0.2, 0.3]])
            
            # Verification score > 0.6 (verified) but different scene types
            mock_scores.side_effect = [
                np.array([[0.75]]),  # Verification score (verified)
                np.array([[0.85, 0.2, 0.1, 0.15]])  # Category scores (real_fight)
            ]
            
            test_image = Image.new('RGB', (100, 100))
            
            result = provider.verify_analysis(
                image=test_image,
                qwen_summary="Organized sport",
                qwen_scene_type="organized_sport",  # Disagrees with Nemotron
                qwen_risk_score=30
            )
            
            # Assertions
            assert result['verified'] is True
            assert result['agreement'] is False  # Different scene types
            assert result['nemotron_scene_type'] == 'real_fight'
            
            # Should use higher risk score (conservative)
            expected_nemotron_score = int(np.mean([80, 95]))  # 87
            expected_recommended = max(30, expected_nemotron_score)  # 87
            
            assert result['recommended_score'] == expected_recommended
            assert result['confidence'] == 0.5
    
    def test_timeout_handling(self):
        """Test that timeout returns fallback using Qwen score"""
        from backend.services.vlm_service import NemotronProvider
        
        provider = NemotronProvider()
        if not provider.available:
            pytest.skip("Nemotron model not available")
        
        with patch.object(provider, 'forward_images') as mock_images:
            # Simulate timeout by raising TimeoutError
            mock_images.side_effect = TimeoutError("Nemotron verification exceeded timeout")
            
            test_image = Image.new('RGB', (100, 100))
            
            result = provider.verify_analysis(
                image=test_image,
                qwen_summary="Fight detected",
                qwen_scene_type="real_fight",
                qwen_risk_score=85,
                timeout=3.0
            )
            
            # Assertions for timeout fallback
            assert result['timed_out'] is True
            assert result['verified'] is False
            assert result['recommended_score'] == 85  # Uses Qwen score
            assert result['confidence'] == 0.6  # Lower confidence
            assert result['nemotron_scene_type'] == "real_fight"  # Uses Qwen scene type
            assert result['agreement'] is True  # Assumes agreement
            assert 'latency_ms' in result
    
    def test_category_score_mapping(self):
        """Test that category scores are correctly mapped to risk ranges"""
        from backend.services.vlm_service import NemotronProvider
        
        provider = NemotronProvider()
        if not provider.available:
            pytest.skip("Nemotron model not available")
        
        test_cases = [
            # (category_scores, expected_scene_type, expected_risk_range)
            ([0.85, 0.2, 0.1, 0.15], 'real_fight', (80, 95)),
            ([0.2, 0.85, 0.1, 0.15], 'organized_sport', (20, 35)),
            ([0.1, 0.2, 0.85, 0.15], 'normal', (10, 25)),
            ([0.1, 0.2, 0.15, 0.85], 'suspicious', (60, 75)),
        ]
        
        for category_scores, expected_scene, expected_range in test_cases:
            with patch.object(provider, 'forward_images') as mock_images, \
                 patch.object(provider, 'forward_queries') as mock_queries, \
                 patch.object(provider, 'get_scores') as mock_scores:
                
                mock_images.return_value = np.array([[0.1, 0.2, 0.3]])
                mock_queries.return_value = np.array([[0.1, 0.2, 0.3]])
                
                mock_scores.side_effect = [
                    np.array([[0.75]]),  # Verification score
                    np.array([category_scores])  # Category scores
                ]
                
                test_image = Image.new('RGB', (100, 100))
                
                result = provider.verify_analysis(
                    image=test_image,
                    qwen_summary="Test",
                    qwen_scene_type=expected_scene,
                    qwen_risk_score=50
                )
                
                assert result['nemotron_scene_type'] == expected_scene
                expected_risk = int(np.mean(expected_range))
                assert result['nemotron_risk_score'] == expected_risk
    
    def test_uncertain_category_score(self):
        """Test that low category scores result in uncertain risk score"""
        from backend.services.vlm_service import NemotronProvider
        
        provider = NemotronProvider()
        if not provider.available:
            pytest.skip("Nemotron model not available")
        
        with patch.object(provider, 'forward_images') as mock_images, \
             patch.object(provider, 'forward_queries') as mock_queries, \
             patch.object(provider, 'get_scores') as mock_scores:
            
            mock_images.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_queries.return_value = np.array([[0.1, 0.2, 0.3]])
            
            # All category scores below 0.7 (uncertain)
            mock_scores.side_effect = [
                np.array([[0.75]]),  # Verification score
                np.array([[0.5, 0.4, 0.45, 0.3]])  # All low category scores
            ]
            
            test_image = Image.new('RGB', (100, 100))
            
            result = provider.verify_analysis(
                image=test_image,
                qwen_summary="Unclear activity",
                qwen_scene_type="normal",
                qwen_risk_score=40
            )
            
            # Should use moderate score of 50 for uncertain classification
            assert result['nemotron_risk_score'] == 50
    
    def test_latency_tracking(self):
        """Test that latency is tracked in milliseconds"""
        from backend.services.vlm_service import NemotronProvider
        
        provider = NemotronProvider()
        if not provider.available:
            pytest.skip("Nemotron model not available")
        
        with patch.object(provider, 'forward_images') as mock_images, \
             patch.object(provider, 'forward_queries') as mock_queries, \
             patch.object(provider, 'get_scores') as mock_scores:
            
            mock_images.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_queries.return_value = np.array([[0.1, 0.2, 0.3]])
            
            mock_scores.side_effect = [
                np.array([[0.75]]),
                np.array([[0.85, 0.2, 0.1, 0.15]])
            ]
            
            test_image = Image.new('RGB', (100, 100))
            
            result = provider.verify_analysis(
                image=test_image,
                qwen_summary="Test",
                qwen_scene_type="real_fight",
                qwen_risk_score=85
            )
            
            # Check latency is present and reasonable
            assert 'latency_ms' in result
            assert isinstance(result['latency_ms'], int)
            assert result['latency_ms'] >= 0
    
    def test_model_unavailable_fallback(self):
        """Test that model unavailable raises RuntimeError"""
        from backend.services.vlm_service import NemotronProvider
        
        # Create provider with unavailable model
        provider = NemotronProvider()
        
        # Force model to be unavailable
        provider.available = False
        
        test_image = Image.new('RGB', (100, 100))
        
        # Should raise RuntimeError when model is unavailable
        with pytest.raises(RuntimeError, match="Nemotron model not available"):
            provider.verify_analysis(
                image=test_image,
                qwen_summary="Fight detected",
                qwen_scene_type="real_fight",
                qwen_risk_score=85
            )
    
    def test_forward_images_unavailable(self):
        """Test that forward_images raises error when model unavailable"""
        from backend.services.vlm_service import NemotronProvider
        
        provider = NemotronProvider()
        provider.available = False
        
        test_image = Image.new('RGB', (100, 100))
        
        with pytest.raises(RuntimeError, match="Nemotron model not available"):
            provider.forward_images([test_image])
    
    def test_forward_queries_unavailable(self):
        """Test that forward_queries raises error when model unavailable"""
        from backend.services.vlm_service import NemotronProvider
        
        provider = NemotronProvider()
        provider.available = False
        
        with pytest.raises(RuntimeError, match="Nemotron model not available"):
            provider.forward_queries(["test query"])
    
    def test_get_scores_unavailable(self):
        """Test that get_scores raises error when model unavailable"""
        from backend.services.vlm_service import NemotronProvider
        
        provider = NemotronProvider()
        provider.available = False
        
        with pytest.raises(RuntimeError, match="Nemotron model not available"):
            provider.get_scores(np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]]))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
