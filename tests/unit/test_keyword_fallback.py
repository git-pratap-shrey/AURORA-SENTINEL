"""
Unit Tests for Keyword-Based Fallback Scoring

Tests the parse_ai_response function in aiRouter_enhanced.py
to ensure proper keyword-based scoring without hardcoded values.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
"""

import pytest
import sys
from pathlib import Path

# Add project root and ai-intelligence-layer to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai-intelligence-layer"))

from aiRouter_enhanced import parse_ai_response


class TestKeywordBasedFallback:
    """Test suite for keyword-based fallback scoring."""
    
    def test_fight_keywords_without_sport_indicators(self):
        """
        Test fight keywords without sport indicators → 75-90 range.
        
        **Validates: Requirements 1.2**
        """
        test_cases = [
            "I see a fight happening with people punching each other",
            "There is violence and aggressive behavior with kicking",
            "An assault is taking place with striking and hitting",
            "Fighting and brawl detected with aggressive actions",
        ]
        
        for response_text in test_cases:
            result = parse_ai_response(response_text, ml_score=50)
            
            # Verify score is in 75-90 range
            assert 75 <= result['aiScore'] <= 90, \
                f"Expected score 75-90, got {result['aiScore']} for: {response_text}"
            
            # Verify scene type is real_fight
            assert result['sceneType'] == 'real_fight', \
                f"Expected 'real_fight', got '{result['sceneType']}' for: {response_text}"
            
            # Verify confidence is reasonable
            assert 0.0 <= result['confidence'] <= 1.0
    
    def test_sport_indicators(self):
        """
        Test sport indicators → 20-35 range.
        
        **Validates: Requirements 1.3**
        """
        test_cases = [
            "Boxing match with protective gear and gloves",
            "Martial arts training with headgear and referee present",
            "Controlled sparring in a ring with protective equipment",
            "Training session with mat and referee supervision",
        ]
        
        for response_text in test_cases:
            result = parse_ai_response(response_text, ml_score=50)
            
            # Verify score is in 20-35 range
            assert 20 <= result['aiScore'] <= 35, \
                f"Expected score 20-35, got {result['aiScore']} for: {response_text}"
            
            # Verify scene type is organized_sport
            assert result['sceneType'] == 'organized_sport', \
                f"Expected 'organized_sport', got '{result['sceneType']}' for: {response_text}"
            
            # Verify confidence is reasonable
            assert 0.0 <= result['confidence'] <= 1.0
    
    def test_suspicious_keywords(self):
        """
        Test suspicious keywords → 60-75 range.
        
        **Validates: Requirements 1.2**
        """
        test_cases = [
            "Crowd surrounding two people with concealment behavior",
            "People concealing others with suspicious behavior",
            "Unknown item detected in the area",
            "Suspicious object with crowd surrounding the area",
        ]
        
        for response_text in test_cases:
            result = parse_ai_response(response_text, ml_score=50)
            
            # Verify score is in 60-75 range
            assert 60 <= result['aiScore'] <= 75, \
                f"Expected score 60-75, got {result['aiScore']} for: {response_text}"
            
            # Verify scene type is suspicious
            assert result['sceneType'] == 'suspicious', \
                f"Expected 'suspicious', got '{result['sceneType']}' for: {response_text}"
            
            # Verify confidence is reasonable
            assert 0.0 <= result['confidence'] <= 1.0
    
    def test_normal_keywords(self):
        """
        Test normal keywords → 10-25 range.
        
        **Validates: Requirements 1.4**
        """
        test_cases = [
            "Normal activity with people walking and talking",
            "Safe environment with peaceful conversation",
            "People standing and talking, no threat detected",
            "Normal safe activity, peaceful scene",
        ]
        
        for response_text in test_cases:
            result = parse_ai_response(response_text, ml_score=50)
            
            # Verify score is in 10-25 range
            assert 10 <= result['aiScore'] <= 25, \
                f"Expected score 10-25, got {result['aiScore']} for: {response_text}"
            
            # Verify scene type is normal
            assert result['sceneType'] == 'normal', \
                f"Expected 'normal', got '{result['sceneType']}' for: {response_text}"
            
            # Verify confidence is reasonable
            assert 0.0 <= result['confidence'] <= 1.0
    
    def test_mixed_keywords_sport_takes_precedence(self):
        """
        Test mixed keywords (fight + sport) → sport takes precedence.
        
        **Validates: Requirements 1.3**
        """
        test_cases = [
            "Fighting with boxing gloves and referee present",
            "Aggressive punching but with protective gear and ring",
            "Martial arts fight with headgear and controlled environment",
            "Sparring match with violence but protective equipment",
        ]
        
        for response_text in test_cases:
            result = parse_ai_response(response_text, ml_score=50)
            
            # Verify sport indicators take precedence
            assert result['sceneType'] == 'organized_sport', \
                f"Expected 'organized_sport' (sport should take precedence), got '{result['sceneType']}' for: {response_text}"
            
            # Verify score is in sport range (20-35)
            assert 20 <= result['aiScore'] <= 35, \
                f"Expected score 20-35 (sport range), got {result['aiScore']} for: {response_text}"
    
    def test_heavy_fighting_keywords(self):
        """
        Test heavy fighting keywords → 80-95 range.
        
        **Validates: Requirements 1.2**
        """
        test_cases = [
            "Multiple strikes with sustained aggression and visible injury",
            "Severe violence with blood and brutal attack",
            "Weapon detected with knife and severe assault",
            "Heavy fighting with multiple strikes and injury",
        ]
        
        for response_text in test_cases:
            result = parse_ai_response(response_text, ml_score=50)
            
            # Verify score is in 80-95 range (higher than regular fight)
            assert 80 <= result['aiScore'] <= 95, \
                f"Expected score 80-95, got {result['aiScore']} for: {response_text}"
            
            # Verify scene type is real_fight
            assert result['sceneType'] == 'real_fight', \
                f"Expected 'real_fight', got '{result['sceneType']}' for: {response_text}"
    
    def test_no_hardcoded_40_score(self):
        """
        Verify no hardcoded 40 score is returned.
        
        **Validates: Requirements 1.5**
        """
        test_cases = [
            "Some random text without clear indicators",
            "Unclear scene description",
            "Ambiguous content",
            "No specific keywords present",
        ]
        
        for response_text in test_cases:
            result = parse_ai_response(response_text, ml_score=50)
            
            # Verify score is NOT 40 (the old hardcoded value)
            assert result['aiScore'] != 40, \
                f"Found hardcoded 40 score for: {response_text}"
            
            # Verify score is in a valid range
            assert 0 <= result['aiScore'] <= 100
    
    def test_no_ml_score_multiplier(self):
        """
        Verify no ml_score * 0.6 multiplier is used.
        
        **Validates: Requirements 1.5**
        """
        # Test with various ML scores
        ml_scores = [30, 50, 70, 90]
        response_text = "Some ambiguous text"
        
        for ml_score in ml_scores:
            result = parse_ai_response(response_text, ml_score)
            
            # Verify score is NOT ml_score * 0.6
            expected_multiplied = ml_score * 0.6
            assert result['aiScore'] != expected_multiplied, \
                f"Found ml_score * 0.6 multiplier: {result['aiScore']} == {expected_multiplied}"
            
            # Verify score is independent of ml_score
            # (keyword analysis should not depend on ml_score)
            assert 0 <= result['aiScore'] <= 100
    
    def test_json_parsing_success(self):
        """
        Test that valid JSON is parsed correctly.
        
        **Validates: Requirements 1.1**
        """
        json_response = '{"aiScore": 85, "sceneType": "real_fight", "explanation": "Fight detected", "confidence": 0.9}'
        
        result = parse_ai_response(json_response, ml_score=50)
        
        assert result['aiScore'] == 85
        assert result['sceneType'] == 'real_fight'
        assert result['explanation'] == 'Fight detected'
        assert result['confidence'] == 0.9
    
    def test_deprecated_scene_types_normalized(self):
        """
        Test that deprecated scene types (drama, prank) are normalized to 'normal'.
        
        **Validates: Requirements 1.1**
        """
        deprecated_types = ['drama', 'prank', 'staged', 'performance']
        
        for scene_type in deprecated_types:
            json_response = f'{{"aiScore": 30, "sceneType": "{scene_type}", "explanation": "Test", "confidence": 0.7}}'
            
            result = parse_ai_response(json_response, ml_score=50)
            
            # Verify deprecated types are converted to 'normal'
            assert result['sceneType'] == 'normal', \
                f"Expected '{scene_type}' to be normalized to 'normal', got '{result['sceneType']}'"
    
    def test_boxing_scene_type_normalized(self):
        """
        Test that 'boxing' scene type is normalized to 'organized_sport'.
        
        **Validates: Requirements 1.1**
        """
        json_response = '{"aiScore": 25, "sceneType": "boxing", "explanation": "Boxing match", "confidence": 0.8}'
        
        result = parse_ai_response(json_response, ml_score=50)
        
        # Verify 'boxing' is converted to 'organized_sport'
        assert result['sceneType'] == 'organized_sport', \
            f"Expected 'boxing' to be normalized to 'organized_sport', got '{result['sceneType']}'"
    
    def test_all_valid_scene_types(self):
        """
        Test that all valid scene types are supported.
        
        **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
        """
        valid_types = ['real_fight', 'organized_sport', 'normal', 'suspicious']
        
        for scene_type in valid_types:
            json_response = f'{{"aiScore": 50, "sceneType": "{scene_type}", "explanation": "Test", "confidence": 0.7}}'
            
            result = parse_ai_response(json_response, ml_score=50)
            
            # Verify valid types are preserved
            assert result['sceneType'] == scene_type, \
                f"Expected '{scene_type}' to be preserved, got '{result['sceneType']}'"
    
    def test_confidence_values(self):
        """
        Test that confidence values are reasonable for different scenarios.
        
        **Validates: Requirements 1.1**
        """
        test_cases = [
            ("Fight with violence", 'real_fight', 0.7, 0.9),
            ("Boxing with gloves", 'organized_sport', 0.6, 0.8),
            ("Normal walking", 'normal', 0.6, 0.8),
            ("Crowd surrounding people with concealment", 'suspicious', 0.5, 0.7),
        ]
        
        for response_text, expected_type, min_conf, max_conf in test_cases:
            result = parse_ai_response(response_text, ml_score=50)
            
            assert result['sceneType'] == expected_type
            assert min_conf <= result['confidence'] <= max_conf, \
                f"Expected confidence {min_conf}-{max_conf}, got {result['confidence']} for: {response_text}"
    
    def test_explanation_preserved(self):
        """
        Test that explanation text is preserved in the result.
        
        **Validates: Requirements 1.1**
        """
        response_text = "This is a detailed explanation of what I see in the image with fight and violence"
        
        result = parse_ai_response(response_text, ml_score=50)
        
        # Verify explanation contains the original text (or truncated version)
        assert 'explanation' in result
        assert len(result['explanation']) > 0
        assert isinstance(result['explanation'], str)
    
    def test_case_insensitive_keyword_matching(self):
        """
        Test that keyword matching is case-insensitive.
        
        **Validates: Requirements 1.2**
        """
        test_cases = [
            "FIGHT with VIOLENCE and AGGRESSION",
            "Fight With Violence And Aggression",
            "fight with violence and aggression",
        ]
        
        results = []
        for response_text in test_cases:
            result = parse_ai_response(response_text, ml_score=50)
            results.append(result)
        
        # All should produce the same scene type and similar scores
        scene_types = [r['sceneType'] for r in results]
        assert len(set(scene_types)) == 1, "Case sensitivity affecting results"
        assert scene_types[0] == 'real_fight'
        
        # All scores should be in the fight range
        for result in results:
            assert 75 <= result['aiScore'] <= 90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
