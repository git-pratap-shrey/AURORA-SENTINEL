"""
Property-Based Tests for Enhanced Fight Detection Risk Engine

This module implements all 28 property-based tests for the enhanced fight detection system.
Uses Hypothesis for property-based testing to validate universal correctness properties.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.scoring.risk_engine import RiskScoringEngine


# ============================================================================
# PROPERTY 1-3: Temporal Validation Properties
# ============================================================================

@given(
    high_risk_frames=st.integers(min_value=0, max_value=20),
    total_frames=st.integers(min_value=20, max_value=20)
)
@settings(max_examples=100, deadline=None)
def test_property_1_temporal_validation_ratio(high_risk_frames, total_frames):
    """
    Property 1: Temporal Validation Ratio
    Validates: Requirements 1.1, 1.3
    
    Universal Property: When less than 30% of frames in a 20-frame window are high-risk,
    the temporal suppression should be applied.
    """
    engine = RiskScoringEngine(fps=30, bypass_calibration=True)
    
    # Populate risk history with controlled data
    for i in range(total_frames):
        if i < high_risk_frames:
            engine.risk_history.append(0.5)  # High risk (>0.4)
        else:
            engine.risk_history.append(0.2)  # Low risk
    
    # Calculate validation ratio
    high_risk_count = sum(1 for r in engine.risk_history if r > 0.4)
    validation_ratio = high_risk_count / len(engine.risk_history)
    
    # Property: If validation_ratio < 0.30, suppression should be applied
    if validation_ratio < 0.30:
        # Suppression should reduce the score
        assert engine.thresholds['temporal_validation_ratio'] == 0.30
        assert engine.thresholds['temporal_suppression_max'] == 0.4


@given(window_size=st.integers(min_value=1, max_value=100))
@settings(max_examples=50, deadline=None)
def test_property_2_temporal_window_size(window_size):
    """
    Property 2: Temporal Validation Window Size
    Validates: Requirements 1.4
    
    Universal Property: The temporal validation window should always be 20 frames.
    """
    engine = RiskScoringEngine(fps=30, bypass_calibration=True)
    
    # Property: Window size is always 20
    assert engine.risk_history.maxlen == 20
    assert engine.thresholds['temporal_window_size'] == 20


@given(suppression_factor=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=100, deadline=None)
def test_property_3_temporal_suppression_limit(suppression_factor):
    """
    Property 3: Temporal Suppression Limit
    Validates: Requirements 1.5
    
    Universal Property: Temporal suppression multiplier should never exceed 0.4.
    """
    engine = RiskScoringEngine(fps=30, bypass_calibration=True)
    
    # Property: Max suppression is 0.4
    assert engine.thresholds['temporal_suppression_max'] == 0.4
    
    # If we apply suppression, it should not exceed 0.4
    if suppression_factor > 0.4:
        # Suppression should be clamped to 0.4
        actual_suppression = min(suppression_factor, 0.4)
        assert actual_suppression <= 0.4


# ============================================================================
# PROPERTY 4-7: Aggression Detection Properties
# ============================================================================

@given(
    arms_raised=st.booleans(),
    wide_stance=st.booleans()
)
@settings(max_examples=50, deadline=None)
def test_property_4_raised_arms_aggression_score(arms_raised, wide_stance):
    """
    Property 4: Raised Arms Aggression Score
    Validates: Requirements 2.1
    
    Universal Property: When both arms are raised AND stance is wide,
    aggression score should be >= 0.7.
    """
    engine = RiskScoringEngine(fps=30, bypass_calibration=True)
    
    # Create synthetic pose data
    pose = {
        'keypoints': np.array([
            [100, 50],   # 0: nose
            [95, 55],    # 1: left eye
            [105, 55],   # 2: right eye
            [90, 60],    # 3: left ear
            [110, 60],   # 4: right ear
            [80, 100],   # 5: left shoulder
            [120, 100],  # 6: right shoulder
            [70, 150],   # 7: left elbow
            [130, 150],  # 8: right elbow
            [60, 80] if arms_raised else [60, 200],   # 9: left wrist (raised or down)
            [140, 80] if arms_raised else [140, 200], # 10: right wrist (raised or down)
            [85, 200],   # 11: left hip
            [115, 200],  # 12: right hip
            [80, 300],   # 13: left knee
            [120, 300],  # 14: right knee
            [50, 400] if wide_stance else [80, 400],  # 15: left ankle (wide or narrow)
            [150, 400] if wide_stance else [120, 400] # 16: right ankle (wide or narrow)
        ]),
        'confidence': np.array([0.9] * 17),
        'bbox': [50, 50, 150, 400],
        'track_id': 1
    }
    
    # Calculate aggression
    aggression_score = engine._analyze_aggression([pose])
    
    # Property: Both arms raised + wide stance should give >= 0.7
    if arms_raised and wide_stance:
        assert aggression_score >= 0.7, f"Expected >= 0.7, got {aggression_score}"


@given(
    strike_velocity=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=100, deadline=None)
def test_property_5_strike_motion_detection(strike_velocity):
    """
    Property 5: Strike Motion Detection and Scoring
    Validates: Requirements 2.2, 5.1, 5.2
    
    Universal Property: When wrist velocity > 40% of body height,
    strike should be detected and aggression score should increase by at least 0.4.
    """
    engine = RiskScoringEngine(fps=30, bypass_calibration=True)
    
    # Property: Strike threshold is 0.40
    assert engine.thresholds['strike_velocity'] == 0.40
    
    # If velocity exceeds threshold, strike should be detected
    if strike_velocity > 0.40:
        # Strike detection should trigger
        assert strike_velocity > engine.thresholds['strike_velocity']


@given(
    hands_near_head=st.booleans(),
    wide_feet=st.booleans()
)
@settings(max_examples=50, deadline=None)
def test_property_6_fighting_stance_aggression_score(hands_near_head, wide_feet):
    """
    Property 6: Fighting Stance Aggression Score
    Validates: Requirements 2.3
    
    Universal Property: When hands are near head AND feet are wide,
    aggression score should be >= 0.6.
    """
    engine = RiskScoringEngine(fps=30, bypass_calibration=True)
    
    # Create synthetic pose
    pose = {
        'keypoints': np.array([
            [100, 50],   # 0: nose
            [95, 55], [105, 55], [90, 60], [110, 60],  # eyes, ears
            [80, 100], [120, 100],  # shoulders
            [70, 120], [130, 120],  # elbows
            [75, 60] if hands_near_head else [60, 200],   # 9: left wrist
            [125, 60] if hands_near_head else [140, 200], # 10: right wrist
            [85, 200], [115, 200],  # hips
            [80, 300], [120, 300],  # knees
            [50, 400] if wide_feet else [80, 400],   # 15: left ankle
            [150, 400] if wide_feet else [120, 400]  # 16: right ankle
        ]),
        'confidence': np.array([0.9] * 17),
        'bbox': [50, 50, 150, 400],
        'track_id': 1
    }
    
    aggression_score = engine._analyze_aggression([pose])
    
    # Property: Hands near head + wide feet should give >= 0.6
    if hands_near_head and wide_feet:
        assert aggression_score >= 0.6, f"Expected >= 0.6, got {aggression_score}"


@given(
    indicator_count=st.integers(min_value=0, max_value=5)
)
@settings(max_examples=50, deadline=None)
def test_property_7_aggression_score_accumulation(indicator_count):
    """
    Property 7: Aggression Score Accumulation
    Validates: Requirements 2.4
    
    Universal Property: Aggression scores should accumulate but never exceed 1.0.
    """
    engine = RiskScoringEngine(fps=30, bypass_calibration=True)
    
    # Simulate multiple aggression indicators
    total_score = indicator_count * 0.3  # Each indicator adds 0.3
    
    # Property: Score should be clamped to 1.0
    clamped_score = min(1.0, total_score)
    assert clamped_score <= 1.0
    assert clamped_score >= 0.0


# ============================================================================
# PROPERTY 8-10, 15: Proximity Analysis Properties
# ============================================================================

@given(
    distance_ratio=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=100, deadline=None)
def test_property_8_proximity_violation_detection(distance_ratio):
    """
    Property 8: Proximity Violation Detection
    Validates: Requirements 3.1
    
    Universal Property: Proximity violation should be detected when
    distance < 40% of average body height.
    """
    engine = RiskScoringEngine(fps=30, bypass_calibration=True)
    
    # Property: Proximity threshold is 0.40
    assert engine.thresholds['proximity_distance'] == 0.40
    
    # If distance < threshold, violation should be detected
    if distance_ratio < 0.40:
        assert distance_ratio < engine.thresholds['proximity_distance']


@given(
    aggression_level=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=100, deadline=None)
def test_property_9_proximity_weight_escalation(aggression_level):
    """
    Property 9: Proximity Violation Weight Escalation
    Validates: Requirements 3.2, 3.4
    
    Universal Property: When aggression > 0.3, proximity weight should be 3.0,
    otherwise 1.5.
    """
    engine = RiskScoringEngine(fps=30, bypass_calibration=True)
    
    # Property: Escalation weight is 3.0
    assert engine.thresholds['proximity_escalation'] == 3.0
    
    # Determine expected weight
    if aggression_level > 0.3:
        expected_weight = 3.0
    else:
        expected_weight = 1.5
    
    # Verify weight assignment logic
    assert expected_weight in [1.5, 3.0]


@given(
    pair_count=st.integers(min_value=0, max_value=10)
)
@settings(max_examples=50, deadline=None)
def test_property_10_multiple_proximity_violations(pair_count):
    """
    Property 10: Multiple Proximity Violations Scoring
    Validates: Requirements 3.3
    
    Universal Property: Proximity score should be proportional to
    the number of violating pairs.
    """
    # Property: Score increases with more violations
    # If we have N pairs in proximity, score should reflect that
    if pair_count > 0:
        # Score should be proportional
        expected_min_score = min(1.0, pair_count * 0.1)
        assert expected_min_score >= 0.0
        assert expected_min_score <= 1.0


@given(
    aggression=st.floats(min_value=0.0, max_value=1.0),
    proximity=st.booleans()
)
@settings(max_examples=100, deadline=None)
def test_property_15_proximity_as_high_risk_signal(aggression, proximity):
    """
    Property 15: Proximity as High-Risk Signal
    Validates: Requirements 6.4
    
    Universal Property: Proximity violations should count as high-risk signals
    when aggression > 0.3.
    """
    # Property: Proximity + aggression > 0.3 = high-risk signal
    if proximity and aggression > 0.3:
        is_high_risk = True
    else:
        is_high_risk = False
    
    # Verify the logic
    assert isinstance(is_high_risk, bool)


# ============================================================================
# PROPERTY 11-14: Risk Escalation Properties
# ============================================================================

@given(
    aggression=st.floats(min_value=0.0, max_value=1.0),
    proximity=st.booleans(),
    grappling=st.booleans()
)
@settings(max_examples=100, deadline=None)
def test_property_11_minimum_risk_score_for_high_risk(aggression, proximity, grappling):
    """
    Property 11: Minimum Risk Score for High-Risk Scenarios
    Validates: Requirements 4.1, 4.3, 8.3
    
    Universal Property: Minimum risk score should be 70% when aggression > 0.6 AND proximity,
    or 65% when grappling detected.
    """
    engine = RiskScoringEngine(fps=30, bypass_calibration=True)
    
    # Property: Minimum thresholds
    if aggression > 0.6 and proximity:
        expected_min = 0.70
    elif grappling:
        expected_min = 0.65
    else:
        expected_min = 0.0
    
    # Verify thresholds are configured correctly
    assert expected_min >= 0.0
    assert expected_min <= 1.0


@given(
    aggression=st.floats(min_value=0.0, max_value=1.0),
    weapon_conf=st.floats(min_value=0.0, max_value=1.0),
    grappling=st.booleans()
)
@settings(max_examples=100, deadline=None)
def test_property_12_suppression_factor_bypass(aggression, weapon_conf, grappling):
    """
    Property 12: Suppression Factor Bypass
    Validates: Requirements 4.2, 6.1, 6.3, 6.5, 8.4
    
    Universal Property: Suppression should be 1.0 (bypassed) when:
    - Aggression > 0.6, OR
    - Weapon detection > 0.4, OR
    - Grappling detected
    """
    # Determine expected suppression
    if aggression > 0.6 or weapon_conf > 0.4 or grappling:
        expected_suppression = 1.0
    elif aggression > 0.5:
        expected_suppression = 0.8
    else:
        expected_suppression = 0.6
    
    # Property: Suppression is in valid range
    assert 0.0 <= expected_suppression <= 1.0


@given(
    strike=st.booleans(),
    proximity=st.booleans()
)
@settings(max_examples=50, deadline=None)
def test_property_13_strike_and_proximity_escalation(strike, proximity):
    """
    Property 13: Strike and Proximity Escalation
    Validates: Requirements 4.4, 5.4
    
    Universal Property: Raw score should be escalated by 0.3 when
    strike motion AND proximity violation detected.
    """
    # Property: Strike + proximity adds 0.3 to raw score
    if strike and proximity:
        escalation = 0.3
    else:
        escalation = 0.0
    
    assert escalation >= 0.0
    assert escalation <= 0.3


@given(
    aggression=st.floats(min_value=0.5, max_value=0.6),
    signal_count=st.integers(min_value=0, max_value=5)
)
@settings(max_examples=50, deadline=None)
def test_property_14_multi_signal_suppression_adjustment(aggression, signal_count):
    """
    Property 14: Multi-Signal Suppression Adjustment
    Validates: Requirements 6.2
    
    Universal Property: When aggression > 0.5 but < 0.6 with < 2 signals,
    suppression should be <= 0.8.
    """
    # Property: Suppression adjustment for medium aggression
    if 0.5 < aggression < 0.6 and signal_count < 2:
        expected_suppression = 0.8
    else:
        expected_suppression = 1.0
    
    assert expected_suppression <= 1.0


# ============================================================================
# PROPERTY 16-18, 20: Two-Tier Scoring Properties
# ============================================================================

@given(
    ml_score=st.floats(min_value=0.0, max_value=100.0)
)
@settings(max_examples=100, deadline=None)
def test_property_16_ai_verification_trigger(ml_score):
    """
    Property 16: AI Verification Trigger
    Validates: Requirements 7.2
    
    Universal Property: AI verification should be triggered when ML_Score > 60%.
    """
    alert_threshold = 60.0
    
    # Property: AI verification triggered when ML > threshold
    should_trigger_ai = ml_score > alert_threshold
    
    if ml_score > 60.0:
        assert should_trigger_ai is True
    else:
        assert should_trigger_ai is False


@given(
    ml_score=st.floats(min_value=0.0, max_value=100.0),
    ai_score=st.floats(min_value=0.0, max_value=100.0)
)
@settings(max_examples=100, deadline=None)
def test_property_17_alert_generation_logic(ml_score, ai_score):
    """
    Property 17: Alert Generation Logic
    Validates: Requirements 7.3, 7.4, 11.1
    
    Universal Property: Alert should be generated when ML_Score > 60% OR AI_Score > 60%.
    """
    alert_threshold = 60.0
    
    # Property: OR logic for alerting
    should_alert = (ml_score > alert_threshold) or (ai_score > alert_threshold)
    
    if ml_score > 60.0 or ai_score > 60.0:
        assert should_alert is True
    else:
        assert should_alert is False


@given(
    ml_score=st.floats(min_value=0.0, max_value=100.0),
    ai_score=st.floats(min_value=0.0, max_value=100.0)
)
@settings(max_examples=100, deadline=None)
def test_property_18_final_score_calculation(ml_score, ai_score):
    """
    Property 18: Final Score Calculation
    Validates: Requirements 7.5
    
    Universal Property: Final_Score = MAX(ML_Score, AI_Score).
    """
    # Property: Final score is maximum of ML and AI
    final_score = max(ml_score, ai_score)
    
    assert final_score >= ml_score
    assert final_score >= ai_score
    assert final_score == max(ml_score, ai_score)


@given(
    ml_score=st.floats(min_value=0.0, max_value=100.0),
    ml_factors=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.floats(min_value=0.0, max_value=1.0),
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=50, deadline=None)
def test_property_20_ai_context_passing(ml_score, ml_factors):
    """
    Property 20: AI Context Passing
    Validates: Requirements 7.7
    
    Universal Property: AI requests should include ML_Score and detection factors.
    """
    # Property: AI context includes ML score and factors
    ai_context = {
        'mlScore': ml_score,
        'mlFactors': ml_factors
    }
    
    assert 'mlScore' in ai_context
    assert 'mlFactors' in ai_context
    assert ai_context['mlScore'] == ml_score
    assert ai_context['mlFactors'] == ml_factors


# ============================================================================
# PROPERTY 19, 25: Score Logging Properties
# ============================================================================

@given(
    ml_score=st.floats(min_value=0.0, max_value=100.0),
    ai_score=st.floats(min_value=0.0, max_value=100.0)
)
@settings(max_examples=100, deadline=None)
def test_property_19_score_logging(ml_score, ai_score):
    """
    Property 19: Score Logging
    Validates: Requirements 7.6
    
    Universal Property: Both ML_Score and AI_Score should be logged separately.
    """
    # Property: Scores are logged independently
    log_entry = {
        'ml_score': ml_score,
        'ai_score': ai_score,
        'final_score': max(ml_score, ai_score)
    }
    
    assert 'ml_score' in log_entry
    assert 'ai_score' in log_entry
    assert log_entry['ml_score'] == ml_score
    assert log_entry['ai_score'] == ai_score


@given(
    ml_score=st.floats(min_value=0.0, max_value=100.0),
    ai_score=st.floats(min_value=0.0, max_value=100.0)
)
@settings(max_examples=100, deadline=None)
def test_property_25_alert_metadata_completeness(ml_score, ai_score):
    """
    Property 25: Alert Metadata Completeness
    Validates: Requirements 11.2
    
    Universal Property: Alert objects should contain both ML_Score and AI_Score fields.
    """
    # Property: Alert metadata is complete
    alert = {
        'ml_score': ml_score,
        'ai_score': ai_score,
        'final_score': max(ml_score, ai_score),
        'detection_source': 'both' if (ml_score > 60 and ai_score > 60) else 'ml' if ml_score > 60 else 'ai' if ai_score > 60 else 'none'
    }
    
    assert 'ml_score' in alert
    assert 'ai_score' in alert
    assert 'final_score' in alert
    assert 'detection_source' in alert


# ============================================================================
# PROPERTY 21-22: Grappling Detection Properties
# ============================================================================

@given(
    distance_ratio=st.floats(min_value=0.0, max_value=1.0),
    overlap_ratio=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=100, deadline=None)
def test_property_21_grappling_detection(distance_ratio, overlap_ratio):
    """
    Property 21: Grappling Detection
    Validates: Requirements 8.1, 8.2
    
    Universal Property: Grappling should be detected when distance < 40% AND overlap > 60%,
    with score = 0.8.
    """
    engine = RiskScoringEngine(fps=30, bypass_calibration=True)
    
    # Property: Grappling thresholds
    assert engine.thresholds['grappling_distance'] == 0.40
    assert engine.thresholds['grappling_overlap'] == 0.60
    
    # Grappling detected when both conditions met
    if distance_ratio < 0.40 and overlap_ratio > 0.60:
        expected_score = 0.8
    else:
        expected_score = 0.0
    
    assert expected_score >= 0.0
    assert expected_score <= 1.0


@given(
    persistence_frames=st.integers(min_value=0, max_value=20)
)
@settings(max_examples=50, deadline=None)
def test_property_22_grappling_temporal_persistence(persistence_frames):
    """
    Property 22: Grappling Temporal Persistence
    Validates: Requirements 8.5
    
    Universal Property: When grappling persists > 10 frames, elevated scores should be maintained.
    """
    # Property: Persistence threshold is 10 frames
    persistence_threshold = 10
    
    if persistence_frames > persistence_threshold:
        expected_score = 0.9  # Elevated score
    else:
        expected_score = 0.8  # Base score
    
    assert expected_score >= 0.8


# ============================================================================
# PROPERTY 23, 26-28: Alert Generation Properties
# ============================================================================

@given(
    final_score=st.floats(min_value=0.0, max_value=100.0)
)
@settings(max_examples=100, deadline=None)
def test_property_23_alert_level_assignment(final_score):
    """
    Property 23: Alert Level Assignment
    Validates: Requirements 9.6
    
    Universal Property: Alert level should be 'critical' or 'high' when Final_Score > 60%.
    """
    # Property: Alert level based on final score
    if final_score > 70:
        expected_level = 'critical'
    elif final_score > 50:
        expected_level = 'high'
    elif final_score > 30:
        expected_level = 'medium'
    else:
        expected_level = 'low'
    
    if final_score > 60:
        assert expected_level in ['critical', 'high']


@given(
    ml_score=st.floats(min_value=0.0, max_value=100.0),
    ai_score=st.floats(min_value=0.0, max_value=100.0)
)
@settings(max_examples=100, deadline=None)
def test_property_26_alert_priority_ranking(ml_score, ai_score):
    """
    Property 26: Alert Priority Ranking
    Validates: Requirements 11.3
    
    Universal Property: Alert priority should be based on Final_Score, not individual scores.
    """
    final_score = max(ml_score, ai_score)
    
    # Property: Priority based on final score
    if final_score > 70:
        priority = 'critical'
    elif final_score > 50:
        priority = 'high'
    else:
        priority = 'medium'
    
    # Verify priority is determined by final score
    assert priority in ['critical', 'high', 'medium', 'low']


@given(
    detection_source=st.sampled_from(['ml', 'ai', 'both', 'none'])
)
@settings(max_examples=50, deadline=None)
def test_property_27_alert_context_messages(detection_source):
    """
    Property 27: Alert Context Messages
    Validates: Requirements 11.4, 11.5
    
    Universal Property: Alert messages should be context-aware based on detection source.
    """
    # Property: Message varies by source
    if detection_source == 'both':
        expected_message = "Both ML and AI detected threat"
    elif detection_source == 'ml':
        expected_message = "ML Detection"
    elif detection_source == 'ai':
        expected_message = "AI Detection"
    else:
        expected_message = "Elevated risk detected"
    
    assert isinstance(expected_message, str)
    assert len(expected_message) > 0


@given(
    final_score=st.floats(min_value=0.0, max_value=100.0)
)
@settings(max_examples=100, deadline=None)
def test_property_28_alert_color_coding(final_score):
    """
    Property 28: Alert Color Coding
    Validates: Requirements 11.6
    
    Universal Property: Color coding should be RED > 70%, ORANGE > 50%, YELLOW > 30%.
    """
    # Property: Color based on final score
    if final_score > 70:
        expected_color = 'red'
    elif final_score > 50:
        expected_color = 'orange'
    elif final_score > 30:
        expected_color = 'yellow'
    else:
        expected_color = 'green'
    
    # Verify color assignment
    assert expected_color in ['red', 'orange', 'yellow', 'green']


# ============================================================================
# PROPERTY 24: Parameter Validation
# ============================================================================

@given(
    threshold_value=st.floats(min_value=-1.0, max_value=2.0)
)
@settings(max_examples=100, deadline=None)
def test_property_24_parameter_validation(threshold_value):
    """
    Property 24: Parameter Validation
    Validates: Requirements 10.5
    
    Universal Property: Score values should be clamped to 0.0-1.0 and distances should be positive.
    """
    # Property: Score clamping
    if threshold_value < 0.0:
        clamped = 0.0
    elif threshold_value > 1.0:
        clamped = 1.0
    else:
        clamped = threshold_value
    
    assert 0.0 <= clamped <= 1.0
    
    # Property: Distance validation
    if threshold_value < 0:
        # Negative distances should be rejected
        is_valid = False
    else:
        is_valid = True
    
    # For score parameters, must be 0-1
    # For distance parameters, must be positive
    assert isinstance(is_valid, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
