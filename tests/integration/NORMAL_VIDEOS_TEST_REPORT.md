# Normal Videos Test Report

**Date**: 2026-03-03  
**Test Suite**: Normal Video Validation  
**Status**: ❌ FAILED - Pass rate below threshold

## Executive Summary

Tested 49 normal videos from `data/sample_videos/Normal_Videos_for_Event_Recognition/` to validate that the two-tier fight detection system produces low risk scores for normal activities.

**Result**: System is producing **too many false positives** on normal videos.

## Test Results

### Overall Statistics

- **Total videos tested**: 49
- **Passed**: 23 (46.9%)
- **Failed**: 26 (53.1%)
- **Pass rate threshold**: 80.0%
- **Status**: ❌ FAILED (46.9% < 80%)

### Score Statistics

#### ML Score Statistics
- **Mean**: 30.9%
- **Max**: 72.7%
- **Min**: 0.0%
- **Std**: 20.3%

#### AI Score Statistics
- **Mean**: 1.0%
- **Max**: 10.0%
- **Min**: 0.0%
- **Std**: 3.0%

#### Final Score Statistics
- **Mean**: 30.9%
- **Max**: 72.7%
- **Min**: 0.0%
- **Std**: 20.3%

## Failed Videos (26/49)

Videos that exceeded the 30% threshold:

1. Normal_Videos_015_x264.mp4: ML=34.9%, AI=0.0%, Final=34.9%
2. Normal_Videos_100_x264.mp4: ML=34.0%, AI=0.0%, Final=34.0%
3. Normal_Videos_129_x264.mp4: ML=48.8%, AI=0.0%, Final=48.8%
4. Normal_Videos_150_x264.mp4: ML=32.2%, AI=0.0%, Final=32.2%
5. Normal_Videos_246_x264.mp4: ML=39.6%, AI=0.0%, Final=39.6%
6. Normal_Videos_247_x264.mp4: ML=40.4%, AI=0.0%, Final=40.4%
7. Normal_Videos_289_x264.mp4: ML=31.1%, AI=0.0%, Final=31.1%
8. Normal_Videos_310_x264.mp4: ML=39.0%, AI=0.0%, Final=39.0%
9. Normal_Videos_312_x264.mp4: ML=35.8%, AI=0.0%, Final=35.8%
10. Normal_Videos_345_x264.mp4: ML=60.6%, AI=10.0%, Final=60.6% ⚠️ ALERT TRIGGERED
11. Normal_Videos_352_x264.mp4: ML=40.9%, AI=0.0%, Final=40.9%
12. Normal_Videos_360_x264.mp4: ML=63.1%, AI=10.0%, Final=63.1% ⚠️ ALERT TRIGGERED
13. Normal_Videos_365_x264.mp4: ML=63.1%, AI=10.0%, Final=63.1% ⚠️ ALERT TRIGGERED
14. Normal_Videos_417_x264.mp4: ML=42.2%, AI=0.0%, Final=42.2%
15. Normal_Videos_439_x264.mp4: ML=49.6%, AI=0.0%, Final=49.6%
16. Normal_Videos_452_x264.mp4: ML=47.9%, AI=0.0%, Final=47.9%
17. Normal_Videos_453_x264.mp4: ML=41.4%, AI=0.0%, Final=41.4%
18. Normal_Videos_478_x264.mp4: ML=40.9%, AI=0.0%, Final=40.9%
19. Normal_Videos_722_x264.mp4: ML=46.0%, AI=0.0%, Final=46.0%
20. Normal_Videos_745_x264.mp4: ML=56.3%, AI=0.0%, Final=56.3%
21. Normal_Videos_758_x264.mp4: ML=71.2%, AI=10.0%, Final=71.2% ⚠️ ALERT TRIGGERED
22. Normal_Videos_781_x264.mp4: ML=72.7%, AI=10.0%, Final=72.7% ⚠️ ALERT TRIGGERED
23. Normal_Videos_798_x264.mp4: ML=52.9%, AI=0.0%, Final=52.9%
24. Normal_Videos_801_x264.mp4: ML=52.8%, AI=0.0%, Final=52.8%
25. Normal_Videos_828_x264.mp4: ML=45.2%, AI=0.0%, Final=45.2%
26. Normal_Videos_929_x264.mp4: ML=37.8%, AI=0.0%, Final=37.8%

**⚠️ CRITICAL**: 5 normal videos triggered alerts (scores > 60%)!

## Root Cause Analysis

### Primary Issues

1. **Grappling Detection Too Sensitive**
   - Detected in 30+ normal videos
   - Triggers on people standing close together
   - Triggers on normal social interactions
   - **Recommendation**: Increase grappling distance threshold or overlap threshold

2. **High Aggression Scores for Normal Movements**
   - Normal arm movements detected as "raised arms"
   - Walking/standing detected as "fighting stance"
   - **Recommendation**: Increase aggression thresholds or add movement pattern analysis

3. **Strike Motion False Positives**
   - Normal hand gestures detected as strikes
   - Walking arm swings detected as strikes
   - **Recommendation**: Increase strike velocity threshold or add context analysis

4. **Proximity Violations on Normal Interactions**
   - People talking/standing together flagged as proximity violations
   - **Recommendation**: Increase proximity distance threshold for normal scenarios

### Debug Log Patterns

Most common false positive triggers:
- `DEBUG: Grappling/Clinching detected` - 200+ occurrences
- `DEBUG: High aggression + proximity detected` - 50+ occurrences
- `DEBUG: Strike motion + proximity detected` - 40+ occurrences
- `DEBUG: Contradiction detected (High Aggression + High Loitering)` - 20+ occurrences

## Passed Videos (23/49)

Videos with scores < 30%:

1. Normal_Videos_050_x264.mp4: ML=1.8%
2. Normal_Videos_248_x264.mp4: ML=25.0%
3. Normal_Videos_251_x264.mp4: ML=24.3%
4. Normal_Videos_317_x264.mp4: ML=15.3%
5. Normal_Videos_401_x264.mp4: ML=28.4%
6. Normal_Videos_576_x264.mp4: ML=5.4%
7. Normal_Videos_597_x264.mp4: ML=27.3%
8. Normal_Videos_603_x264.mp4: ML=26.4%
9. Normal_Videos_606_x264.mp4: ML=4.2%
10. Normal_Videos_621_x264.mp4: ML=12.9%
11. Normal_Videos_641_x264.mp4: ML=21.2%
12. Normal_Videos_656_x264.mp4: ML=16.4%
13. Normal_Videos_696_x264.mp4: ML=4.3%
14. Normal_Videos_704_x264.mp4: ML=4.4%
15. Normal_Videos_831_x264.mp4: ML=2.4%
16. Normal_Videos_877_x264.mp4: ML=0.9%
17. Normal_Videos_881_x264.mp4: ML=24.9%
18. Normal_Videos_885_x264.mp4: ML=23.8%
19. Normal_Videos_892_x264.mp4: ML=13.2%
20. Normal_Videos_905_x264.mp4: ML=9.6%
21. Normal_Videos_912_x264.mp4: ML=0.0%
22. Normal_Videos_913_x264.mp4: ML=0.0%
23. Normal_Videos_914_x264.mp4: ML=0.0%

## Recommendations

### Immediate Actions Required

1. **Calibrate Grappling Detection**
   - Current: distance < 40% height AND overlap > 60%
   - Recommended: distance < 30% height AND overlap > 70%
   - Or add temporal persistence requirement (sustained for 5+ frames)

2. **Increase Aggression Thresholds**
   - Current: raised_arms=0.7, strike=0.5, fighting_stance=0.6
   - Recommended: raised_arms=0.8, strike=0.6, fighting_stance=0.7
   - Or add velocity/acceleration analysis to distinguish fighting from normal movement

3. **Increase Strike Velocity Threshold**
   - Current: 40% of body height per frame
   - Recommended: 50-60% of body height per frame
   - Or add directional analysis (strikes toward another person)

4. **Adjust Proximity Thresholds**
   - Current: 40% of average height
   - Recommended: 30% of average height for high-risk escalation
   - Keep 40% for baseline detection but reduce escalation weight

5. **Add Context-Aware Suppression**
   - Implement "loitering + aggression contradiction" more aggressively
   - Add "slow movement" detection to suppress false positives
   - Add "social interaction" pattern detection

### Testing Strategy

1. Re-test with calibrated thresholds
2. Target: 80%+ pass rate (40/49 videos)
3. Acceptable: Some false positives on edge cases (people hugging, high-fiving, etc.)
4. Critical: Zero false positives on clearly normal activities (walking, standing, talking)

## Conclusion

The ML layer is **too aggressive** and needs calibration. While the design goal is to detect ANY fighting-like behavior, the current thresholds are triggering on normal social interactions and movements.

**Status**: ❌ CALIBRATION REQUIRED

**Next Steps**:
1. Adjust thresholds in `config/risk_thresholds.yaml`
2. Re-run normal video tests
3. Validate fight video detection still works
4. Iterate until 80%+ pass rate achieved

---

**Test File**: `tests/integration/test_normal_videos.py`  
**Generated**: 2026-03-03
