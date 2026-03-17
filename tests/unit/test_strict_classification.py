"""
Unit Tests for Strict Classification in Qwen2-VL Integration

Tests the _parse_response method in qwen2vl_integration.py
to ensure strict classification rules are enforced and deprecated
scene types (prank, drama) are never returned.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**
"""

import pytest
import sys
from pathlib import Path

# Add project root and ai-intelligence-layer to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai-intelligence-layer"))

from qwen2vl_integration import Qwen2VLAnalyzer


class TestStrictClassification:
    """Test suite for strict classification rules in Qwen2-VL."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing (without loading the model)."""
        # We'll test the _parse_response method directly without initializing the full model
        analyzer = object.__new__(Qwen2VLAnalyzer)
        return analyzer
    
    def test_prank_never_returned_json(self, analyzer):
        """
        Test that 'prank' is never returned as sceneType (JSON parsing).
        
        **Validates: Requirements 2.1, 2.2**
        """
        test_cases = [
            '{"aiScore": 30, "sceneType": "prank", "explanation": "Prank fight", "confidence": 0.7}',
            '{"aiScore": 25, "sceneType": "prank", "explanation": "Staged prank", "confidence": 0.6}',
        ]
        
        for response_text in test_cases:
            result = analyzer._parse_response(response_text)
            
            # Verify 'prank' is converted to 'normal'
            assert result['sceneType'] != 'prank', \
                f"'prank' should never be returned, got: {result['sceneType']}"
            assert result['sceneType'] == 'normal', \
                f"Expected 'prank' to be normalized to 'normal', got: {result['sceneType']}"
    
    def test_drama_never_returned_json(self, analyzer):
        """
        Test that 'drama' is never returned as sceneType (JSON parsing).
        
        **Validates: Requirements 2.1, 2.2**
        """
        test_cases = [
            '{"aiScore": 35, "sceneType": "drama", "explanation": "Dramatic scene", "confidence": 0.7}',
            '{"aiScore": 28, "sceneType": "drama", "explanation": "Drama performance", "confidence": 0.6}',
        ]
        
        for response_text in test_cases:
            result = analyzer._parse_response(response_text)
            
            # Verify 'drama' is converted to 'normal'
            assert result['sceneType'] != 'drama', \
                f"'drama' should never be returned, got: {result['sceneType']}"
            assert result['sceneType'] == 'normal', \
                f"Expected 'drama' to be normalized to 'normal', got: {result['sceneType']}"
    
    def test_all_deprecated_types_normalized(self, analyzer):
        """
        Test that all deprecated scene types are normalized to 'normal'.
        
        **Validates: Requirements 2.1, 2.2**
        """
        deprecated_types = ['prank', 'drama', 'staged', 'performance']
        
        for scene_type in deprecated_types:
            json_response = f'{{"aiScore": 30, "sceneType": "{scene_type}", "explanation": "Test", "confidence": 0.7}}'
            
            result = analyzer._parse_response(json_response)
            
            # Verify deprecated types are never returned
            assert result['sceneType'] != scene_type, \
                f"Deprecated type '{scene_type}' should never be returned"
            assert result['sceneType'] == 'normal', \
                f"Expected '{scene_type}' to be normalized to 'normal', got '{result['sceneType']}'"
    
    def test_only_valid_scene_types_returned(self, analyzer):
        """
        Test that only valid scene types are returned.
        
        **Validates: Requirements 2.2**
        """
        valid_types = ['real_fight', 'organized_sport', 'normal', 'suspicious']
        
        # Test with various response texts
        test_cases = [
            "Fight with violence and aggression",
            "Boxing with protective gear and referee",
            "Normal walking and talking",
            "Crowd surrounding people with concealment",
            "Some random text",
            '{"aiScore": 50, "sceneType": "real_fight", "explanation": "Test", "confidence": 0.7}',
        ]
        
        for response_text in test_cases:
            result = analyzer._parse_response(response_text)
            
            # Verify only valid types are returned
            assert result['sceneType'] in valid_types, \
                f"Invalid scene type '{result['sceneType']}' returned for: {response_text}"
    
    def test_fight_without_sport_indicators_real_fight(self, analyzer):
        """
        Test fight without sport indicators → real_fight.
        
        **Validates: Requirements 2.3**
        """
        test_cases = [
            "Physical aggression with punching and kicking",
            "Fight detected with violence and assault",
            "Aggressive behavior with hitting and striking",
            "Attack in progress with fighting",
        ]
        
        for response_text in test_cases:
            result = analyzer._parse_response(response_text)
            
            # Verify scene type is real_fight
            assert result['sceneType'] == 'real_fight', \
                f"Expected 'real_fight' for fight without sport indicators, got '{result['sceneType']}' for: {response_text}"
            
            # Verify score is in fight range (75-90)
            assert 75 <= result['aiScore'] <= 90, \
                f"Expected score 75-90, got {result['aiScore']} for: {response_text}"
    
    def test_fight_with_protective_gear_organized_sport(self, analyzer):
        """
        Test fight with protective gear + referee → organized_sport (capped at 35).
        
        **Validates: Requirements 2.4, 2.5**
        """
        test_cases = [
            "Boxing match with protective gear and gloves, referee present",
            "Martial arts with headgear and referee in the ring",
            "Sparring with protective equipment and referee supervision",
            "Training with gloves, mat, and referee",
        ]
        
        for response_text in test_cases:
            result = analyzer._parse_response(response_text)
            
            # Verify scene type is organized_sport
            assert result['sceneType'] == 'organized_sport', \
                f"Expected 'organized_sport' for fight with protective gear, got '{result['sceneType']}' for: {response_text}"
            
            # Verify score is capped at 35 (20-35 range)
            assert 20 <= result['aiScore'] <= 35, \
                f"Expected score 20-35 (capped), got {result['aiScore']} for: {response_text}"
    
    def test_heavy_fighting_keywords_high_range(self, analyzer):
        """
        Test heavy fighting keywords → 80-95 range.
        
        **Validates: Requirements 2.6**
        """
        test_cases = [
            "Multiple strikes with sustained aggression and visible injury",
            "Severe violence with blood and brutal attack",
            "Heavy fighting with multiple strikes and injury",
            "Sustained aggression with visible injury and blood",
        ]
        
        for response_text in test_cases:
            result = analyzer._parse_response(response_text)
            
            # Verify scene type is real_fight
            assert result['sceneType'] == 'real_fight', \
                f"Expected 'real_fight' for heavy fighting, got '{result['sceneType']}' for: {response_text}"
            
            # Verify score is in heavy fighting range (80-95)
            assert 80 <= result['aiScore'] <= 95, \
                f"Expected score 80-95 for heavy fighting, got {result['aiScore']} for: {response_text}"
    
    def test_suspicious_crowd_behavior(self, analyzer):
        """
        Test suspicious crowd behavior → suspicious (60-75).
        
        **Validates: Requirements 2.7**
        """
        test_cases = [
            "Crowd surrounding two people with concealment behavior",
            "Unknown item detected in suspicious context",
            "Suspicious object with crowd surrounding the area",
            "Suspicious behavior detected in the scene",
        ]
        
        for response_text in test_cases:
            result = analyzer._parse_response(response_text)
            
            # Verify scene type is suspicious
            assert result['sceneType'] == 'suspicious', \
                f"Expected 'suspicious' for crowd behavior, got '{result['sceneType']}' for: {response_text}"
            
            # Verify score is in suspicious range (60-75)
            assert 60 <= result['aiScore'] <= 75, \
                f"Expected score 60-75 for suspicious behavior, got {result['aiScore']} for: {response_text}"
    
    def test_sport_indicators_override_fight_keywords(self, analyzer):
        """
        Test that sport indicators override fight keywords.
        
        **Validates: Requirements 2.3, 2.4**
        """
        test_cases = [
            "Fighting with boxing gloves and referee present",
            "Aggressive punching but with protective gear and ring",
            "Martial arts fight with headgear and controlled environment",
            "Violence but with protective equipment and referee",
        ]
        
        for response_text in test_cases:
            result = analyzer._parse_response(response_text)
            
            # Verify sport indicators take precedence
            assert result['sceneType'] == 'organized_sport', \
                f"Expected 'organized_sport' (sport should override fight), got '{result['sceneType']}' for: {response_text}"
            
            # Verify score is in sport range (20-35), not fight range
            assert 20 <= result['aiScore'] <= 35, \
                f"Expected score 20-35 (sport range), got {result['aiScore']} for: {response_text}"
    
    def test_json_parsing_preserves_valid_types(self, analyzer):
        """
        Test that JSON parsing preserves valid scene types.
        
        **Validates: Requirements 2.2**
        """
        valid_types = ['real_fight', 'organized_sport', 'normal', 'suspicious']
        
        for scene_type in valid_types:
            json_response = f'{{"aiScore": 50, "sceneType": "{scene_type}", "explanation": "Test", "confidence": 0.7}}'
            
            result = analyzer._parse_response(json_response)
            
            # Verify valid types are preserved
            assert result['sceneType'] == scene_type, \
                f"Expected '{scene_type}' to be preserved, got '{result['sceneType']}'"
    
    def test_boxing_normalized_to_organized_sport(self, analyzer):
        """
        Test that 'boxing' scene type is normalized to 'organized_sport'.
        
        **Validates: Requirements 2.2, 2.4**
        """
        json_response = '{"aiScore": 25, "sceneType": "boxing", "explanation": "Boxing match", "confidence": 0.8}'
        
        result = analyzer._parse_response(json_response)
        
        # Verify 'boxing' is converted to 'organized_sport'
        assert result['sceneType'] == 'organized_sport', \
            f"Expected 'boxing' to be normalized to 'organized_sport', got '{result['sceneType']}'"
    
    def test_normal_activity_low_score(self, analyzer):
        """
        Test that normal activity gets low scores (10-25).
        
        **Validates: Requirements 2.3**
        """
        test_cases = [
            "Normal activity with people walking and talking",
            "Safe environment with peaceful conversation",
            "People standing and talking, no threat detected",
            "Normal safe activity, peaceful scene",
        ]
        
        for response_text in test_cases:
            result = analyzer._parse_response(response_text)
            
            # Verify scene type is normal
            assert result['sceneType'] == 'normal', \
                f"Expected 'normal', got '{result['sceneType']}' for: {response_text}"
            
            # Verify score is in normal range (10-25)
            assert 10 <= result['aiScore'] <= 25, \
                f"Expected score 10-25, got {result['aiScore']} for: {response_text}"
    
    def test_parsing_method_included(self, analyzer):
        """
        Test that parsing_method is included in the result.
        
        **Validates: Requirements 2.1**
        """
        # Test JSON parsing
        json_response = '{"aiScore": 50, "sceneType": "normal", "explanation": "Test", "confidence": 0.7}'
        result = analyzer._parse_response(json_response)
        assert 'parsing_method' in result
        assert result['parsing_method'] == 'json'
        
        # Test keyword parsing
        keyword_response = "Fight with violence"
        result = analyzer._parse_response(keyword_response)
        assert 'parsing_method' in result
        assert result['parsing_method'] == 'keyword'
    
    def test_confidence_values_reasonable(self, analyzer):
        """
        Test that confidence values are reasonable (0-1 range).
        
        **Validates: Requirements 2.1**
        """
        test_cases = [
            "Fight with violence",
            "Boxing with gloves and referee",
            "Normal walking",
            "Crowd surrounding people",
            "Some random text",
        ]
        
        for response_text in test_cases:
            result = analyzer._parse_response(response_text)
            
            # Verify confidence is in valid range
            assert 0.0 <= result['confidence'] <= 1.0, \
                f"Expected confidence 0.0-1.0, got {result['confidence']} for: {response_text}"
    
    def test_provider_field_present(self, analyzer):
        """
        Test that provider field is present in the result.
        
        **Validates: Requirements 2.1**
        """
        test_cases = [
            '{"aiScore": 50, "sceneType": "normal", "explanation": "Test", "confidence": 0.7}',
            "Fight with violence",
        ]
        
        for response_text in test_cases:
            result = analyzer._parse_response(response_text)
            
            # Verify provider field is present
            assert 'provider' in result
            assert result['provider'] == 'qwen2vl'
    
    def test_explanation_field_present(self, analyzer):
        """
        Test that explanation field is present and non-empty.
        
        **Validates: Requirements 2.1**
        """
        test_cases = [
            '{"aiScore": 50, "sceneType": "normal", "explanation": "Test explanation", "confidence": 0.7}',
            "Fight with violence and aggression",
        ]
        
        for response_text in test_cases:
            result = analyzer._parse_response(response_text)
            
            # Verify explanation is present and non-empty
            assert 'explanation' in result
            assert len(result['explanation']) > 0
            assert isinstance(result['explanation'], str)
    
    def test_case_insensitive_keyword_matching(self, analyzer):
        """
        Test that keyword matching is case-insensitive.
        
        **Validates: Requirements 2.3**
        """
        test_cases = [
            "FIGHT with VIOLENCE and AGGRESSION",
            "Fight With Violence And Aggression",
            "fight with violence and aggression",
        ]
        
        results = []
        for response_text in test_cases:
            result = analyzer._parse_response(response_text)
            results.append(result)
        
        # All should produce the same scene type
        scene_types = [r['sceneType'] for r in results]
        assert len(set(scene_types)) == 1, "Case sensitivity affecting results"
        assert scene_types[0] == 'real_fight'
        
        # All scores should be in the fight range
        for result in results:
            assert 75 <= result['aiScore'] <= 90
    
    def test_no_hardcoded_scores(self, analyzer):
        """
        Test that no hardcoded scores (like 40) are returned.
        
        **Validates: Requirements 2.1**
        """
        test_cases = [
            "Some random text without clear indicators",
            "Unclear scene description",
            "Ambiguous content",
        ]
        
        for response_text in test_cases:
            result = analyzer._parse_response(response_text)
            
            # Verify score is NOT 40 (the old hardcoded value)
            assert result['aiScore'] != 40, \
                f"Found hardcoded 40 score for: {response_text}"
            
            # Verify score is in a valid range
            assert 0 <= result['aiScore'] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
