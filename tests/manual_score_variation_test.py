"""
Manual test to verify AI scores are varied and not hardcoded
Tests that scores vary based on keyword count
"""
import sys
import os

# Add ai-intelligence-layer to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai-intelligence-layer'))

from aiRouter_enhanced import parse_ai_response


def test_score_variation():
    """Test that scores vary based on keyword count"""
    
    print("=" * 60)
    print("TESTING SCORE VARIATION (NO HARDCODED SCORES)")
    print("=" * 60)
    
    # Test fight scenarios with different keyword counts
    test_cases = [
        ("One person is fighting another", "1 fight keyword"),
        ("Two people are fighting and punching each other violently", "3 fight keywords"),
        ("Aggressive fighting with punching, kicking, and striking behavior", "5 fight keywords"),
        ("Multiple strikes with sustained aggression and visible injury", "Heavy fight keywords"),
        ("Boxing match with protective gear and referee present", "Sport keywords"),
        ("Crowd surrounding two people with suspicious behavior", "Suspicious keywords"),
        ("Normal walking and talking, safe activity", "Normal keywords"),
    ]
    
    print("\nTest Results:")
    print("-" * 60)
    
    scores_seen = set()
    
    for text, description in test_cases:
        result = parse_ai_response(text, ml_score=70)
        score = result['aiScore']
        scene = result['sceneType']
        scores_seen.add(score)
        
        print(f"\n{description}:")
        print(f"  Text: {text}")
        print(f"  Score: {score}")
        print(f"  Scene: {scene}")
        print(f"  Confidence: {result['confidence']}")
    
    print("\n" + "=" * 60)
    print(f"UNIQUE SCORES GENERATED: {len(scores_seen)}")
    print(f"Scores: {sorted(scores_seen)}")
    print("=" * 60)
    
    # Verify we have varied scores (not just 3-4 hardcoded values)
    if len(scores_seen) >= 5:
        print("\n✅ SUCCESS: Scores are properly varied!")
        print("   No longer clustering around hardcoded values (40, 75, 83, etc.)")
    else:
        print("\n⚠️  WARNING: Limited score variation detected")
        print(f"   Only {len(scores_seen)} unique scores generated")
    
    # Check that we don't have the old hardcoded values
    old_hardcoded = {40, 83, 88, 68, 28, 18}
    found_old = scores_seen.intersection(old_hardcoded)
    
    if found_old:
        print(f"\n⚠️  Note: Some old mid-range values still present: {found_old}")
        print("   This is OK if they're calculated, not hardcoded")
    else:
        print("\n✅ No old hardcoded mid-range values detected")
    
    return len(scores_seen) >= 5


if __name__ == '__main__':
    success = test_score_variation()
    sys.exit(0 if success else 1)
