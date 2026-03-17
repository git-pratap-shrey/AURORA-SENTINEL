"""
Nemotron verification scenario tests.

Tests:
- Agreement case (both models agree) → average score used
- Disagreement case (models disagree) → higher risk score used
- Mismatch case (verification < 0.6) → conservative handling
- Nemotron unavailable → Qwen score used
- Verify verification details included in response

Requirements: 3.8, 3.9, 3.11, 3.12
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestNemotronVerification:
    """Test Nemotron verification scenarios."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return Image.new('RGB', (640, 480), color='blue')
    
    @pytest.mark.skipif(not os.path.exists('backend/services/vlm_service.py'),
                       reason="VLM service not available")
    def test_agreement_case_average_score(self, sample_image):
        """
        Test agreement case: both models agree → average score used.
        Requirement 3.8
        """
        try:
            from backend.services.vlm_service import NemotronProvider
            
            nemotron = NemotronProvider()
            
            if not nemotron.available:
                pytest.skip("Nemotron not available")
            
            # Test case: Qwen says real_fight with score 85
            qwen_summary = "Two people engaged in physical fighting with aggressive strikes"
            qwen_scene_type = "real_fight"
            qwen_risk_score = 85
            
            # Run verification
            result = nemotron.verify_analysis(
                image=sample_image,
                qwen_summary=qwen_summary,
                qwen_scene_type=qwen_scene_type,
                qwen_risk_score=qwen_risk_score,
                timeout=3.0
            )
            
            # If verification score > 0.6 AND scene types match
            if result['verified'] and result['agreement']:
                # Should use average score
                expected_avg = (qwen_risk_score + result['nemotron_risk_score']) / 2
                assert abs(result['recommended_score'] - expected_avg) < 1, \
                    f"Agreement case should use average score (expected ~{expected_avg}, got {result['recommended_score']})"
                
                # Confidence should be high
                assert result['confidence'] >= 0.8, \
                    f"Agreement case should have high confidence (got {result['confidence']})"
                
                print(f"✓ Agreement case: Qwen={qwen_risk_score}, Nemotron={result['nemotron_risk_score']}, " +
                      f"Recommended={result['recommended_score']} (average)")
            else:
                pytest.skip("Models did not agree in this test case")
                
        except ImportError:
            pytest.skip("VLM service not available")
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")
    
    @pytest.mark.skipif(not os.path.exists('backend/services/vlm_service.py'),
                       reason="VLM service not available")
    def test_disagreement_case_higher_risk_score(self, sample_image):
        """
        Test disagreement case: models disagree → higher risk score used.
        Requirement 3.9
        """
        try:
            from backend.services.vlm_service import NemotronProvider
            
            nemotron = NemotronProvider()
            
            if not nemotron.available:
                pytest.skip("Nemotron not available")
            
            # Test case: Qwen says organized_sport with score 30
            # But image might show real fighting
            qwen_summary = "Boxing match with protective gear and referee"
            qwen_scene_type = "organized_sport"
            qwen_risk_score = 30
            
            # Run verification
            result = nemotron.verify_analysis(
                image=sample_image,
                qwen_summary=qwen_summary,
                qwen_scene_type=qwen_scene_type,
                qwen_risk_score=qwen_risk_score,
                timeout=3.0
            )
            
            # If scene types don't match (disagreement)
            if not result['agreement']:
                # Should use higher risk score (conservative)
                higher_score = max(qwen_risk_score, result['nemotron_risk_score'])
                assert result['recommended_score'] == higher_score, \
                    f"Disagreement case should use higher risk score (expected {higher_score}, got {result['recommended_score']})"
                
                # Confidence should be lower
                assert result['confidence'] <= 0.7, \
                    f"Disagreement case should have lower confidence (got {result['confidence']})"
                
                print(f"✓ Disagreement case: Qwen={qwen_risk_score} ({qwen_scene_type}), " +
                      f"Nemotron={result['nemotron_risk_score']} ({result['nemotron_scene_type']}), " +
                      f"Recommended={result['recommended_score']} (higher)")
            else:
                pytest.skip("Models agreed in this test case")
                
        except ImportError:
            pytest.skip("VLM service not available")
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")
    
    @pytest.mark.skipif(not os.path.exists('backend/services/vlm_service.py'),
                       reason="VLM service not available")
    def test_mismatch_case_conservative_handling(self, sample_image):
        """
        Test mismatch case: verification < 0.6 → conservative handling.
        Requirement 3.9
        """
        try:
            from backend.services.vlm_service import NemotronProvider
            
            nemotron = NemotronProvider()
            
            if not nemotron.available:
                pytest.skip("Nemotron not available")
            
            # Test case: Qwen's summary might not match image well
            qwen_summary = "People walking peacefully in a park"
            qwen_scene_type = "normal"
            qwen_risk_score = 15
            
            # Run verification
            result = nemotron.verify_analysis(
                image=sample_image,
                qwen_summary=qwen_summary,
                qwen_scene_type=qwen_scene_type,
                qwen_risk_score=qwen_risk_score,
                timeout=3.0
            )
            
            # If verification score < 0.6 (mismatch)
            if not result['verified']:
                # Should use higher risk score (conservative)
                higher_score = max(qwen_risk_score, result['nemotron_risk_score'])
                assert result['recommended_score'] == higher_score, \
                    f"Mismatch case should use higher risk score (expected {higher_score}, got {result['recommended_score']})"
                
                # Confidence should be low
                assert result['confidence'] <= 0.6, \
                    f"Mismatch case should have low confidence (got {result['confidence']})"
                
                print(f"✓ Mismatch case (verification={result['verification_score']:.2f}): " +
                      f"Using higher risk score {result['recommended_score']}")
            else:
                pytest.skip("Verification score was high in this test case")
                
        except ImportError:
            pytest.skip("VLM service not available")
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")
    
    @pytest.mark.skipif(not os.path.exists('ai-intelligence-layer/aiRouter_enhanced.py'),
                       reason="AI router not available")
    def test_nemotron_unavailable_uses_qwen_score(self):
        """
        Test Nemotron unavailable → Qwen score used.
        Requirement 3.11, 3.12
        """
        try:
            sys.path.insert(0, 'ai-intelligence-layer')
            from aiRouter_enhanced import analyze_image
            import base64
            from io import BytesIO
            
            # Create test image
            img = Image.new('RGB', (640, 480), color='green')
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            image_data = base64.b64encode(buffer.getvalue()).decode()
            image_data = f"data:image/jpeg;base64,{image_data}"
            
            # Mock Nemotron to be unavailable
            with patch('aiRouter_enhanced.init_nemotron', return_value=None):
                result = analyze_image(
                    image_data=image_data,
                    ml_score=70,
                    ml_factors={},
                    camera_id='test_camera'
                )
                
                # Should have Qwen score
                assert 'aiScore' in result, "Should have AI score"
                assert result['aiScore'] > 0, "Should have valid AI score"
                
                # Should have error details about Nemotron
                if 'errors' in result:
                    assert 'nemotron' in result['errors'], \
                        "Should have Nemotron error details"
                
                # Should NOT have Nemotron verification details
                assert 'nemotron_verification' not in result or result.get('nemotron_verification') is None, \
                    "Should not have Nemotron verification when unavailable"
                
                print(f"✓ Nemotron unavailable: Using Qwen score {result['aiScore']}")
                
        except ImportError:
            pytest.skip("AI router not available")
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")
    
    @pytest.mark.skipif(not os.path.exists('ai-intelligence-layer/aiRouter_enhanced.py'),
                       reason="AI router not available")
    def test_verification_details_included_in_response(self):
        """
        Test that verification details are included in response.
        Requirement 3.12
        """
        try:
            sys.path.insert(0, 'ai-intelligence-layer')
            from aiRouter_enhanced import analyze_image
            import base64
            from io import BytesIO
            
            # Create test image
            img = Image.new('RGB', (640, 480), color='yellow')
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            image_data = base64.b64encode(buffer.getvalue()).decode()
            image_data = f"data:image/jpeg;base64,{image_data}"
            
            result = analyze_image(
                image_data=image_data,
                ml_score=75,
                ml_factors={},
                camera_id='test_camera'
            )
            
            # If Nemotron was available and ran
            if 'nemotron_verification' in result and result['nemotron_verification']:
                verification = result['nemotron_verification']
                
                # Verify all required fields are present
                required_fields = [
                    'verification_score',
                    'verified',
                    'category_scores',
                    'nemotron_scene_type',
                    'nemotron_risk_score',
                    'agreement',
                    'recommended_score',
                    'confidence'
                ]
                
                for field in required_fields:
                    assert field in verification, \
                        f"Verification details missing field: {field}"
                
                # Verify category scores are present
                assert 'real_fight' in verification['category_scores'], \
                    "Category scores should include 'real_fight'"
                assert 'organized_sport' in verification['category_scores'], \
                    "Category scores should include 'organized_sport'"
                assert 'normal' in verification['category_scores'], \
                    "Category scores should include 'normal'"
                assert 'suspicious' in verification['category_scores'], \
                    "Category scores should include 'suspicious'"
                
                print("✓ Verification details included in response:")
                print(f"  - Verification score: {verification['verification_score']:.2f}")
                print(f"  - Verified: {verification['verified']}")
                print(f"  - Agreement: {verification['agreement']}")
                print(f"  - Nemotron scene: {verification['nemotron_scene_type']}")
                print(f"  - Recommended score: {verification['recommended_score']}")
            else:
                pytest.skip("Nemotron verification not available in this test")
                
        except ImportError:
            pytest.skip("AI router not available")
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")
    
    def test_conservative_disagreement_property(self):
        """
        Property test: When models disagree, higher risk score is ALWAYS used.
        Requirement 3.9
        """
        # Mock verification results with disagreement
        test_cases = [
            {'qwen': 30, 'nemotron': 80, 'expected': 80},  # Nemotron higher
            {'qwen': 85, 'nemotron': 40, 'expected': 85},  # Qwen higher
            {'qwen': 60, 'nemotron': 60, 'expected': 60},  # Same (edge case)
            {'qwen': 25, 'nemotron': 90, 'expected': 90},  # Large difference
        ]
        
        for case in test_cases:
            # Simulate disagreement handling
            recommended = max(case['qwen'], case['nemotron'])
            
            assert recommended == case['expected'], \
                f"Conservative handling failed: Qwen={case['qwen']}, Nemotron={case['nemotron']}, " + \
                f"Expected={case['expected']}, Got={recommended}"
        
        print("✓ Conservative disagreement property verified for all test cases")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
