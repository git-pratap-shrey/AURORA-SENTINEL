# Implementation Plan: Enhanced Fight Detection

## Overview

This implementation plan converts the two-tier fight detection design into actionable coding tasks. The system separates ML-based aggressive pattern detection from AI-based context verification, using OR logic for fail-safe alerting. Each task builds incrementally, with property-based tests validating universal correctness properties throughout.

## Tasks

- [x] 1. Set up enhanced risk engine configuration and thresholds
  - Update `models/scoring/risk_engine.py` to add new threshold parameters
  - Add temporal_validation_ratio: 0.30, temporal_window_size: 20, temporal_suppression_max: 0.4
  - Add proximity_distance: 0.40, proximity_escalation: 3.0
  - Add strike_velocity: 0.40, grappling_distance: 0.40, grappling_overlap: 0.60
  - Add aggression thresholds: raised_arms: 0.7, strike: 0.5, fighting_stance: 0.6
  - Initialize keypoint_history tracking structure using defaultdict with deque
  - _Requirements: 10.1, 10.2, 10.3_

- [x]* 1.1 Write property test for parameter validation
  - **Property 24: Parameter Validation**
  - **Validates: Requirements 10.5**
  - Generate random threshold values, verify score values clamped to 0.0-1.0 and distances are positive
  - _Requirements: 10.5_

- [x] 2. Implement enhanced temporal validation
  - [x] 2.1 Update temporal validator to use 20-frame window
    - Modify `_apply_temporal_validation` method in `RiskScoringEngine`
    - Change window size from 30 to 20 frames
    - Update validation ratio from 0.50 to 0.30
    - Change suppression multiplier max from 0.6 to 0.4
    - _Requirements: 1.1, 1.3, 1.4, 1.5_

  - [x]* 2.2 Write property tests for temporal validation
    - **Property 1: Temporal Validation Ratio**
    - **Validates: Requirements 1.1, 1.3**
    - Generate random video sequences with 20+ frames, verify 30% threshold enforcement
    - **Property 2: Temporal Validation Window Size**
    - **Validates: Requirements 1.4**
    - Verify 20-frame minimum window used for all sequences
    - **Property 3: Temporal Suppression Limit**
    - **Validates: Requirements 1.5**
    - Verify suppression multiplier never exceeds 0.4
    - _Requirements: 1.1, 1.3, 1.4, 1.5_

- [x] 3. Implement strike velocity detection
  - [x] 3.1 Add keypoint velocity tracking
    - Create `_update_keypoint_history` method in `RiskScoringEngine`
    - Track wrist and ankle positions per person using track_id
    - Store last 10 positions in deque for each keypoint
    - Normalize positions by person body height
    - _Requirements: 5.3, 5.5_

  - [x] 3.2 Implement strike motion detection
    - Create `_detect_strike_velocity` method in `RiskScoringEngine`
    - Calculate per-frame displacement for wrists and ankles
    - Detect strike when displacement > 40% of body height
    - Return strike indicators per person with velocity magnitude
    - _Requirements: 5.1, 5.2_

  - [x]* 3.3 Write property test for strike detection
    - **Property 5: Strike Motion Detection and Scoring**
    - **Validates: Requirements 2.2, 5.1, 5.2**
    - Generate random pose sequences with controlled wrist velocities
    - Verify strike detection when velocity > 40% body height
    - Verify aggression score increases by at least 0.4
    - _Requirements: 2.2, 5.1, 5.2_

- [x] 4. Implement enhanced aggression detection
  - [x] 4.1 Update aggression scoring for raised arms
    - Modify `_analyze_aggression` method in `RiskScoringEngine`
    - Detect both arms raised above shoulders AND feet spread > 30% body height
    - Assign aggression score of 0.7 for this pose
    - Remove any discrimination logic for controlled movements
    - _Requirements: 2.1, 2.5_

  - [x] 4.2 Add fighting stance detection
    - Detect hands within 25% of body height from head
    - Detect feet spread wider than 30% of body height
    - Assign aggression score of 0.6 for this pose
    - _Requirements: 2.3_

  - [x] 4.3 Integrate strike motion into aggression scoring
    - Call `_detect_strike_velocity` from `_analyze_aggression`
    - Add 0.5 to aggression score when strike detected
    - Add additional 0.4 if arm is extended during strike
    - _Requirements: 2.2_

  - [x] 4.4 Implement aggression score accumulation
    - Sum all aggression indicators per person
    - Clamp total aggression score to maximum of 1.0
    - Return per-person aggression scores
    - _Requirements: 2.4_

  - [x]* 4.5 Write property tests for aggression detection
    - **Property 4: Raised Arms Aggression Score**
    - **Validates: Requirements 2.1**
    - Generate poses with raised arms and wide stance, verify score >= 0.7
    - **Property 6: Fighting Stance Aggression Score**
    - **Validates: Requirements 2.3**
    - Generate poses with hands near head and wide feet, verify score >= 0.6
    - **Property 7: Aggression Score Accumulation**
    - **Validates: Requirements 2.4**
    - Generate poses with multiple indicators, verify accumulation up to 1.0
    - _Requirements: 2.1, 2.3, 2.4_

- [x] 5. Implement enhanced proximity analysis
  - [x] 5.1 Update proximity violation detection
    - Modify `_analyze_proximity` method in `RiskScoringEngine`
    - Change distance threshold to 40% of average body height
    - Calculate center-to-center distance using bbox centers
    - Normalize by average person height
    - _Requirements: 3.1, 3.5_

  - [x] 5.2 Implement proximity weight escalation
    - Check if either person has aggression score > 0.3
    - Apply escalation weight of 3.0 if aggression present
    - Apply baseline weight of 1.5 if no aggression
    - _Requirements: 3.2, 3.4_

  - [x] 5.3 Handle multiple proximity violations
    - Calculate weighted sum of all proximity violations in frame
    - Return score proportional to violation count
    - Count proximity violations as high-risk signals when aggression > 0.3
    - _Requirements: 3.3, 6.4_

  - [x]* 5.4 Write property tests for proximity analysis
    - **Property 8: Proximity Violation Detection**
    - **Validates: Requirements 3.1**
    - Generate random person pairs with controlled distances, verify detection at 40% threshold
    - **Property 9: Proximity Violation Weight Escalation**
    - **Validates: Requirements 3.2, 3.4**
    - Verify weight = 3.0 when aggression > 0.3, else 1.5
    - **Property 10: Multiple Proximity Violations Scoring**
    - **Validates: Requirements 3.3**
    - Generate frames with multiple pairs, verify proportional scoring
    - **Property 15: Proximity as High-Risk Signal**
    - **Validates: Requirements 6.4**
    - Verify proximity violations count as high-risk signals when aggression > 0.3
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 6.4_

- [x] 6. Implement grappling detection
  - [x] 6.1 Create grappling detection method
    - Add `_detect_grappling` method to `RiskScoringEngine`
    - Check if distance < 40% of average height
    - Calculate bounding box overlap using IoU
    - Detect grappling when overlap > 60%
    - Assign grappling factor score of 0.8
    - _Requirements: 8.1, 8.2_

  - [x] 6.2 Implement grappling temporal persistence
    - Track grappling state per person pair using track_ids
    - Maintain elevated risk when grappling persists > 10 frames
    - Store grappling history in instance variable
    - _Requirements: 8.5_

  - [x]* 6.3 Write property tests for grappling detection
    - **Property 21: Grappling Detection**
    - **Validates: Requirements 8.1, 8.2**
    - Generate person pairs with controlled distance and overlap, verify detection and score = 0.8
    - **Property 22: Grappling Temporal Persistence**
    - **Validates: Requirements 8.5**
    - Generate sequences with sustained grappling, verify elevated scores maintained
    - _Requirements: 8.1, 8.2, 8.5_

- [x] 7. Implement risk score escalation and suppression bypass
  - [x] 7.1 Update risk score calculation with minimum thresholds
    - Modify `calculate_risk` method in `RiskScoringEngine`
    - Apply minimum risk score of 70% when aggression > 0.6 AND proximity violation
    - Apply minimum risk score of 65% when grappling detected
    - Escalate raw score by 0.3 when strike motion + proximity violation
    - _Requirements: 4.1, 4.3, 4.4_

  - [x] 7.2 Implement suppression factor bypass logic
    - Set suppression factor to 1.0 when aggression > 0.6
    - Set suppression factor to 1.0 when weapon detection > 0.4
    - Set suppression factor to 1.0 when grappling detected
    - Apply max 0.8 suppression when aggression > 0.5 but < 0.6
    - _Requirements: 4.2, 6.1, 6.2, 6.3, 6.5, 8.4_

  - [x]* 7.3 Write property tests for risk escalation
    - **Property 11: Minimum Risk Score for High-Risk Scenarios**
    - **Validates: Requirements 4.1, 4.3, 8.3**
    - Generate detections with high aggression + proximity or grappling, verify minimum scores
    - **Property 12: Suppression Factor Bypass**
    - **Validates: Requirements 4.2, 6.1, 6.3, 6.5, 8.4**
    - Verify suppression = 1.0 for high aggression, weapons, or grappling
    - **Property 13: Strike and Proximity Escalation**
    - **Validates: Requirements 4.4, 5.4**
    - Verify raw score escalation by 0.3 when strike + proximity
    - **Property 14: Multi-Signal Suppression Adjustment**
    - **Validates: Requirements 6.2**
    - Verify suppression <= 0.8 when aggression > 0.5 with < 2 signals
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 6.1, 6.2, 6.3, 6.5, 8.3, 8.4_

- [x] 8. Checkpoint - Ensure ML risk engine tests pass
  - Run all property-based tests for risk engine components
  - Verify aggression detection, proximity analysis, grappling detection working correctly
  - Ensure all tests pass, ask the user if questions arise

- [x] 9. Create two-tier scoring service
  - [x] 9.1 Create scoring service module
    - Create new file `backend/services/scoring_service.py`
    - Define `TwoTierScoringService` class with risk_engine and ai_client dependencies
    - Set alert_threshold to 60.0
    - _Requirements: 7.1, 7.2_

  - [x] 9.2 Implement calculate_scores method
    - Create async `calculate_scores` method accepting frame, detection_data, context
    - Call risk_engine.calculate_risk to get ML_Score and ml_factors
    - Trigger AI verification if ML_Score > 60%
    - Calculate Final_Score = MAX(ML_Score, AI_Score)
    - Determine detection_source: "ml" | "ai" | "both" | "none"
    - Return ScoringResult with all scores and metadata
    - _Requirements: 7.2, 7.3, 7.4, 7.5_

  - [x]* 9.3 Write property tests for score aggregation
    - **Property 16: AI Verification Trigger**
    - **Validates: Requirements 7.2**
    - Verify AI verification triggered when ML_Score > 60%
    - **Property 17: Alert Generation Logic**
    - **Validates: Requirements 7.3, 7.4, 11.1**
    - Verify alert generated when ML_Score > 60% OR AI_Score > 60%
    - **Property 18: Final Score Calculation**
    - **Validates: Requirements 7.5**
    - Generate random ML and AI scores, verify Final_Score = MAX(ML, AI)
    - _Requirements: 7.2, 7.3, 7.4, 7.5, 11.1_

- [x] 10. Enhance AI intelligence layer
  - [x] 10.1 Update AI router input schema
    - Modify `ai-intelligence-layer/aiRouter.js`
    - Update `analyzeImage` flow inputSchema to include mlScore and mlFactors
    - Add mlScore: z.number() and mlFactors: z.object({})
    - _Requirements: 7.7_

  - [x] 10.2 Update AI router output schema
    - Update outputSchema to include aiScore, explanation, sceneType, confidence
    - Define sceneType enum: "real_fight" | "boxing" | "drama" | "normal"
    - _Requirements: 7.1_

  - [x] 10.3 Enhance AI verification prompt
    - Update system prompt to include ML context
    - Instruct AI to provide objective threat assessment
    - Request AI to distinguish real threats from controlled activities
    - Include ML_Score and ml_factors in prompt context
    - _Requirements: 7.7_

  - [x]* 10.4 Write property test for AI context passing
    - **Property 20: AI Context Passing**
    - **Validates: Requirements 7.7**
    - Verify AI requests include ML_Score and detection factors
    - _Requirements: 7.7_

- [x] 11. Implement score logging
  - [x] 11.1 Add ML/AI score fields to Alert model
    - Update `backend/db/models.py` Alert model
    - Add ml_score: float, ai_score: float, final_score: float fields
    - Add detection_source: str field
    - Add ai_explanation: str and ai_scene_type: str fields
    - _Requirements: 7.6, 11.2_

  - [x] 11.2 Update database migration
    - Create Alembic migration for new Alert fields
    - Apply migration to add columns to alerts table
    - _Requirements: 7.6_

  - [x]* 11.3 Write property tests for score logging
    - **Property 19: Score Logging**
    - **Validates: Requirements 7.6**
    - Verify both ML_Score and AI_Score logged separately
    - **Property 25: Alert Metadata Completeness**
    - **Validates: Requirements 11.2**
    - Verify alert objects contain both ML_Score and AI_Score fields
    - _Requirements: 7.6, 11.2_

- [x] 12. Enhance alert generation service
  - [x] 12.1 Update alert service to use two-tier scores
    - Modify `backend/services/alert_service.py`
    - Update `generate_alert` method to accept ScoringResult
    - Extract ml_score, ai_score, final_score from scoring result
    - _Requirements: 11.1, 11.2_

  - [x] 12.2 Implement alert level and color assignment
    - Determine alert level based on Final_Score
    - Assign 'critical' and 'red' when Final_Score > 70%
    - Assign 'high' and 'orange' when Final_Score > 50%
    - Assign 'medium' and 'yellow' when Final_Score > 30%
    - _Requirements: 11.6_

  - [x] 12.3 Generate context-aware alert messages
    - Create message based on detection_source
    - "Both ML and AI detected threat" when source = "both"
    - "ML Detection - AI Verification: {scene_type}" when source = "ml" and AI_Score < 30%
    - "AI Detection - Review ML Factors" when source = "ai"
    - _Requirements: 11.4, 11.5_

  - [x] 12.4 Populate alert with complete metadata
    - Include ml_score, ai_score, final_score in alert object
    - Include ml_factors, ai_explanation, ai_scene_type
    - Include detection_source, level, color, message
    - _Requirements: 11.2, 11.3_

  - [x]* 12.5 Write property tests for alert generation
    - **Property 23: Alert Level Assignment**
    - **Validates: Requirements 9.6**
    - Verify alert level 'critical' or 'high' when Final_Score > 60%
    - **Property 26: Alert Priority Ranking**
    - **Validates: Requirements 11.3**
    - Verify priority based on Final_Score not individual scores
    - **Property 27: Alert Context Messages**
    - **Validates: Requirements 11.4, 11.5**
    - Verify correct messages for different detection sources
    - **Property 28: Alert Color Coding**
    - **Validates: Requirements 11.6**
    - Verify RED > 70%, ORANGE > 50%, YELLOW > 30%
    - _Requirements: 9.6, 11.3, 11.4, 11.5, 11.6_

- [x] 13. Integrate two-tier scoring into video processing pipeline
  - [x] 13.1 Update video router to use scoring service
    - Modify `backend/api/routers/video.py`
    - Instantiate TwoTierScoringService with risk_engine and ai_client
    - Replace direct risk_engine calls with scoring_service.calculate_scores
    - Pass frame, detection_data, and context to scoring service
    - _Requirements: 7.1_

  - [x] 13.2 Update alert generation calls
    - Pass ScoringResult to alert_service.generate_alert
    - Store alerts with complete two-tier metadata
    - Log both ML and AI scores to database
    - _Requirements: 7.6, 11.1_

  - [x]* 13.3 Write integration tests for video pipeline
    - Test end-to-end flow: detection → ML scoring → AI verification → alert
    - Verify both scores logged correctly in database
    - Verify alerts generated with correct metadata
    - _Requirements: 7.1, 7.6, 11.1_

- [x] 14. Checkpoint - Ensure backend integration tests pass
  - Run integration tests for two-tier scoring pipeline
  - Verify ML and AI scores calculated correctly
  - Verify alerts generated with proper metadata
  - Ensure all tests pass, ask the user if questions arise

- [x] 15. Update frontend to display dual scores
  - [x] 15.1 Update alert display component
    - Modify alert card component to show ml_score and ai_score separately
    - Display Final_Score prominently as primary indicator
    - Show detection_source badge ("ML", "AI", or "Both")
    - Apply color coding based on Final_Score
    - _Requirements: 11.2, 11.3, 11.6_

  - [x] 15.2 Add AI explanation tooltip
    - Display ai_explanation on hover or expand
    - Show ai_scene_type as badge or label
    - Include context message based on detection_source
    - _Requirements: 11.4, 11.5_

  - [x] 15.3 Implement alert filtering controls
    - Add filter dropdowns for ML_Score, AI_Score, Final_Score thresholds
    - Allow operators to filter by detection_source
    - Update alert list based on filter selections
    - _Requirements: 11.7_

  - [x]* 15.4 Write unit tests for frontend components
    - Test alert card rendering with dual scores
    - Test color coding logic
    - Test filter controls functionality
    - _Requirements: 11.2, 11.3, 11.6, 11.7_

- [x] 16. Test with real video files
  - [x] 16.1 Create test script for video validation
    - Create `tests/test_video_validation.py`
    - Load test videos from storage/temp/
    - Process each video through complete pipeline
    - Capture ML_Score, AI_Score, Final_Score for each frame
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 16.2 Validate fight video detection
    - Test `storage/temp/raw_1772268166.802879_4rth.mp4`
    - Verify maximum ML_Score > 70%
    - Verify alert generated with level 'critical' or 'high'
    - _Requirements: 9.1, 9.6_

  - [x] 16.3 Validate second fight video detection
    - Test `storage/temp/raw_1772268210.415873_5th.mp4`
    - Verify maximum ML_Score > 70%
    - Verify alert generated with level 'critical' or 'high'
    - _Requirements: 9.2, 9.6_

  - [x] 16.4 Validate boxing video detection
    - Test `storage/temp/raw_1772268625.365338_scr9-231238~2sdddsd.mp4`
    - Verify ML_Score > 60% (detects fighting poses)
    - Verify AI_Score < 30% (identifies controlled sparring)
    - Verify Final_Score triggers alert for operator review
    - _Requirements: 9.3, 9.4, 9.5_

  - [x]* 16.5 Write unit tests for video validation
    - Test video loading and frame extraction
    - Test detection pipeline integration
    - Test score calculation for each video
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 17. Add error handling and logging
  - [x] 17.1 Implement ML risk engine error handling
    - Handle missing pose data gracefully (return 0.0 score)
    - Skip poses with < 11 valid keypoints or confidence < 0.35
    - Handle tracking loss and missing track_ids
    - Validate person height > 0 before normalization
    - Use frame count as fallback if timestamp unavailable
    - _Requirements: 10.4_

  - [x] 17.2 Implement AI layer error handling
    - Add timeout handling for AI requests (5s timeout)
    - Return AI_Score = 0 with explanation on timeout
    - Implement fallback to ML_Score only if AI unavailable
    - Handle invalid AI responses gracefully
    - Implement exponential backoff for rate limiting
    - Retry once on network errors, then fail gracefully
    - _Requirements: 7.2_

  - [x] 17.3 Implement score aggregation error handling
    - Use 0.0 as default for missing scores
    - Clamp scores to 0-100 range
    - Use thread-safe operations for concurrent access
    - Log scores to file if database write fails
    - _Requirements: 7.5, 7.6_

  - [x] 17.4 Implement alert generation error handling
    - Generate alerts with available data if metadata incomplete
    - Generate alerts without frame image if extraction fails
    - Queue alerts for retry on notification failure
    - Don't block processing pipeline on alert errors
    - _Requirements: 11.1_

  - [x]* 17.5 Write unit tests for error handling
    - Test each error scenario with appropriate inputs
    - Verify graceful degradation
    - Verify error logging
    - _Requirements: 7.2, 7.5, 7.6, 10.4, 11.1_

- [x] 18. Add configuration and threshold documentation
  - [x] 18.1 Create configuration file for thresholds
    - Create `config/risk_thresholds.yaml`
    - Document all threshold parameters with descriptions
    - Include default values and acceptable ranges
    - Add comments explaining each parameter's purpose
    - _Requirements: 10.1, 10.2, 10.3_

  - [x] 18.2 Update risk engine to load from config
    - Modify `RiskScoringEngine.__init__` to accept config file path
    - Load thresholds from YAML file
    - Validate threshold values on load
    - Log loaded threshold values
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [x]* 18.3 Write unit tests for configuration loading
    - Test config file parsing
    - Test threshold validation
    - Test invalid config handling
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 19. Final checkpoint - Comprehensive testing
  - Run all property-based tests (28 properties)
  - Run all unit tests
  - Run all integration tests
  - Test with all three real video files
  - Verify frontend displays dual scores correctly
  - Verify alert filtering works as expected
  - Ensure all tests pass, ask the user if questions arise

- [x] 20. Performance optimization and validation
  - [x] 20.1 Optimize ML scoring performance
    - Profile risk engine to identify bottlenecks
    - Ensure ML scoring completes in < 50ms per frame
    - Optimize keypoint velocity calculations
    - Cache repeated calculations where possible
    - _Requirements: 7.1_

  - [x] 20.2 Optimize AI verification performance
    - Implement async AI requests to avoid blocking
    - Ensure AI verification completes in < 2s per frame
    - Add request queuing to handle burst loads
    - _Requirements: 7.2_

  - [x] 20.3 Validate real-time processing capability
    - Test system with 30 FPS video stream
    - Verify parallel processing doesn't block ML pipeline
    - Measure end-to-end latency from frame to alert
    - _Requirements: 7.1, 7.2_

  - [x]* 20.4 Write performance tests
    - Test ML scoring latency with various frame counts
    - Test AI verification latency
    - Test concurrent request handling
    - _Requirements: 7.1, 7.2_

- [ ] 21. End-to-end integration testing with real videos
  - [ ] 21.1 Create comprehensive integration test suite
    - Create `tests/integration/test_two_tier_integration.py`
    - Test complete pipeline: video ingestion → detection → ML scoring → AI verification → alert generation
    - Verify data flows correctly through all components
    - Test with all three existing test videos
    - _Requirements: 7.1, 7.2, 7.3, 9.1, 9.2, 9.3_

  - [ ] 21.2 Test VLM and ML score correlation
    - Process each test video and capture frame-by-frame ML and AI scores
    - Calculate correlation metrics between ML_Score and AI_Score
    - Identify frames where scores diverge significantly (>30% difference)
    - Generate correlation report with statistics and visualizations
    - _Requirements: 7.5, 7.6, 9.4, 9.5_

  - [ ] 21.3 Validate alert generation with dual scores
    - Verify alerts generated when ML_Score > 60% (AI_Score = 0)
    - Verify alerts generated when AI_Score > 60% (ML_Score < 60%)
    - Verify alerts generated when both scores > 60%
    - Verify no alerts when both scores < 60%
    - Verify detection_source field correctly set ("ml", "ai", "both", "none")
    - _Requirements: 7.3, 7.4, 11.1, 11.2_

  - [ ] 21.4 Test AI context passing and verification
    - Verify AI requests include ML_Score and ml_factors in context
    - Verify AI responses include aiScore, explanation, sceneType, confidence
    - Test AI timeout handling (mock 5s+ delay)
    - Test AI unavailable scenario (mock service down)
    - Verify system falls back to ML_Score only when AI fails
    - _Requirements: 7.2, 7.7_

  - [ ]* 21.5 Write integration test assertions
    - Assert all alerts contain ml_score, ai_score, final_score fields
    - Assert Final_Score = MAX(ML_Score, AI_Score) for all frames
    - Assert alert metadata includes ai_explanation and ai_scene_type
    - Assert color coding matches Final_Score thresholds
    - _Requirements: 7.5, 7.6, 11.2, 11.6_

- [ ] 22. VLM intelligence validation and edge case testing
  - [ ] 22.1 Test VLM scene type classification
    - Process boxing video and verify AI identifies "boxing" or "controlled sparring"
    - Process fight videos and verify AI identifies "real_fight" or high threat
    - Create test cases for drama/staged scenes (if available)
    - Verify AI confidence scores align with scene clarity
    - _Requirements: 7.1, 9.3, 9.4_

  - [ ] 22.2 Test VLM explanation quality
    - Verify AI explanations are non-empty and contextually relevant
    - Check explanations reference visible scene elements
    - Verify explanations distinguish between controlled vs uncontrolled activity
    - Test explanation consistency across similar frames
    - _Requirements: 7.7, 11.4, 11.5_

  - [ ] 22.3 Test conflicting score scenarios
    - Create test cases where ML_Score is high but AI_Score is low (boxing)
    - Create test cases where AI_Score is high but ML_Score is low (if possible)
    - Verify alert messages correctly indicate detection source
    - Verify operators receive context for manual review
    - _Requirements: 11.4, 11.5_

  - [ ] 22.4 Test AI timeout and error handling
    - Mock AI service timeout (>5s response time)
    - Verify system returns AI_Score = 0 with "timeout" explanation
    - Verify alert still generated if ML_Score > 60%
    - Verify processing pipeline continues without blocking
    - Test AI rate limiting and exponential backoff
    - _Requirements: 7.2_

  - [ ]* 22.5 Write unit tests for VLM edge cases
    - Test malformed AI responses
    - Test missing AI fields (aiScore, explanation, sceneType)
    - Test AI score out of range (negative, >100)
    - Test network errors and retry logic
    - _Requirements: 7.2, 7.7_

- [ ] 23. Performance testing under load
  - [ ] 23.1 Test ML scoring performance
    - Create performance test script `tests/performance/test_ml_performance.py`
    - Process 1000 frames and measure ML scoring latency per frame
    - Verify ML scoring completes in < 50ms per frame (95th percentile)
    - Profile risk engine to identify bottlenecks
    - Test with varying numbers of detected persons (1, 2, 5, 10)
    - _Requirements: 7.1_

  - [ ] 23.2 Test AI verification performance
    - Measure AI verification latency for 100 frames
    - Verify AI verification completes in < 2s per frame (95th percentile)
    - Test async request handling to ensure non-blocking
    - Measure impact of AI verification on overall pipeline throughput
    - _Requirements: 7.2_

  - [ ] 23.3 Test concurrent processing capability
    - Simulate multiple camera feeds (3-5 concurrent streams)
    - Verify ML scoring continues at full speed during AI verification
    - Test request queuing and parallel AI verification
    - Measure system resource usage (CPU, memory, GPU)
    - Verify no frame drops or processing delays
    - _Requirements: 7.1, 7.2_

  - [ ] 23.4 Test real-time processing at 30 FPS
    - Process 30 FPS video stream for 60 seconds (1800 frames)
    - Measure end-to-end latency from frame ingestion to alert generation
    - Verify system maintains real-time processing without backlog
    - Test with high-activity video (many detections per frame)
    - _Requirements: 7.1, 7.2_

  - [ ]* 23.5 Write performance benchmark tests
    - Create automated performance regression tests
    - Set performance thresholds and fail tests if exceeded
    - Generate performance reports with latency distributions
    - _Requirements: 7.1, 7.2_

- [ ] 24. Database and logging validation
  - [ ] 24.1 Test dual score logging to database
    - Process test videos and verify all alerts saved to database
    - Query alerts table and verify ml_score, ai_score, final_score fields populated
    - Verify detection_source, ai_explanation, ai_scene_type fields present
    - Test database migration applied correctly
    - _Requirements: 7.6, 11.2_

  - [ ] 24.2 Test alert metadata completeness
    - Verify all alerts contain complete ml_factors dictionary
    - Verify all alerts contain timestamp, camera_id, frame_number
    - Test alert retrieval and filtering by score thresholds
    - Verify alert priority ranking uses Final_Score
    - _Requirements: 11.2, 11.3_

  - [ ] 24.3 Test score logging error handling
    - Mock database write failure
    - Verify scores logged to file as fallback
    - Verify processing pipeline continues despite logging errors
    - Test alert retry queue for failed notifications
    - _Requirements: 7.6, 11.1_

  - [ ]* 24.4 Write database integration tests
    - Test alert CRUD operations
    - Test score filtering queries
    - Test database connection pooling under load
    - _Requirements: 7.6, 11.2_

- [ ] 25. Frontend integration and operator workflow testing
  - [ ] 25.1 Test dual score display in alert cards
    - Verify alert cards show ml_score and ai_score separately
    - Verify Final_Score displayed prominently
    - Verify detection_source badge shows "ML", "AI", or "Both"
    - Test color coding: RED (>70%), ORANGE (>50%), YELLOW (>30%)
    - _Requirements: 11.2, 11.3, 11.6_

  - [ ] 25.2 Test AI explanation display
    - Verify ai_explanation visible on hover or expand
    - Verify ai_scene_type displayed as badge
    - Test context messages for different detection sources
    - Verify "ML Detection - AI Verification: {scene_type}" when ML high, AI low
    - Verify "AI Detection - Review ML Factors" when AI high, ML low
    - _Requirements: 11.4, 11.5_

  - [ ] 25.3 Test alert filtering controls
    - Test filter by ML_Score threshold (slider or input)
    - Test filter by AI_Score threshold
    - Test filter by Final_Score threshold
    - Test filter by detection_source ("ml", "ai", "both")
    - Verify alert list updates correctly based on filters
    - _Requirements: 11.7_

  - [ ] 25.4 Test operator workflow with real videos
    - Process all three test videos through complete system
    - Verify operators can see both scores for each alert
    - Verify operators can filter alerts by score type
    - Test alert acknowledgment and dismissal
    - Verify forensic review includes both ML and AI context
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 11.7_

  - [ ]* 25.5 Write frontend component tests
    - Test alert card rendering with various score combinations
    - Test filter control state management
    - Test color coding logic
    - Test tooltip and expand/collapse functionality
    - _Requirements: 11.2, 11.3, 11.6, 11.7_

- [ ] 26. Edge case and error scenario testing
  - [ ] 26.1 Test missing or invalid detection data
    - Test frames with no pose detections (ML_Score = 0)
    - Test frames with invalid keypoints (< 11 valid points)
    - Test frames with low confidence poses (< 0.35)
    - Test frames with missing track_ids
    - Verify system handles gracefully without crashes
    - _Requirements: 10.4_

  - [ ] 26.2 Test score boundary conditions
    - Test ML_Score exactly at 60% (should trigger AI)
    - Test ML_Score at 59.9% (should not trigger AI)
    - Test Final_Score at alert thresholds (30%, 50%, 70%)
    - Test score clamping (negative scores, scores > 100)
    - _Requirements: 7.2, 7.3, 7.5_

  - [ ] 26.3 Test temporal edge cases
    - Test video sequences shorter than 20 frames
    - Test sequences with all high-risk frames (100% validation ratio)
    - Test sequences with exactly 30% high-risk frames (boundary)
    - Test tracking loss and recovery scenarios
    - _Requirements: 1.1, 1.3, 1.4_

  - [ ] 26.4 Test concurrent score calculation
    - Test thread-safe score aggregation with concurrent requests
    - Test race conditions in keypoint history tracking
    - Test grappling state tracking with concurrent person pairs
    - Verify no data corruption under concurrent load
    - _Requirements: 7.5_

  - [ ]* 26.5 Write edge case unit tests
    - Test each error handling path
    - Test boundary conditions for all thresholds
    - Test concurrent access patterns
    - _Requirements: 7.2, 7.5, 10.4_

- [ ] 27. System integration validation with chat, latest, and recent features
  - [ ] 27.1 Test chat feature with two-tier scoring
    - Verify chat interface can query alerts by ML_Score, AI_Score, Final_Score
    - Test natural language queries: "Show me alerts where AI disagreed with ML"
    - Test queries: "Show boxing detections" (ML high, AI low)
    - Verify chat responses include both scores and context
    - _Requirements: 11.2, 11.7_

  - [ ] 27.2 Test latest alerts feature
    - Verify latest alerts endpoint returns dual scores
    - Test sorting by Final_Score (default)
    - Test sorting by ML_Score or AI_Score
    - Verify latest alerts include detection_source and ai_explanation
    - _Requirements: 11.2, 11.3_

  - [ ] 27.3 Test recent alerts feature
    - Verify recent alerts timeline shows both scores
    - Test filtering recent alerts by detection_source
    - Test time-based queries with score thresholds
    - Verify recent alerts display color-coded by Final_Score
    - _Requirements: 11.2, 11.6, 11.7_

  - [ ] 27.4 Test alert search and forensic review
    - Test searching alerts by camera_id with score filters
    - Test searching by time range with ML/AI score thresholds
    - Verify forensic review shows complete ml_factors and ai_explanation
    - Test exporting alerts with dual score metadata
    - _Requirements: 7.6, 11.2_

  - [ ]* 27.5 Write integration tests for feature interactions
    - Test chat queries with score filters
    - Test latest/recent endpoints with various filters
    - Test alert search with complex criteria
    - _Requirements: 11.2, 11.7_

- [ ] 28. Configuration and threshold validation
  - [ ] 28.1 Test threshold configuration loading
    - Verify risk engine loads thresholds from `config/risk_thresholds.yaml`
    - Test threshold validation (scores 0.0-1.0, distances positive)
    - Test invalid config handling (malformed YAML, out-of-range values)
    - Verify threshold values logged on startup
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [ ] 28.2 Test threshold modification and reload
    - Modify threshold values in config file
    - Test hot-reload or restart with new thresholds
    - Verify new thresholds applied to subsequent frames
    - Test threshold changes don't affect in-flight processing
    - _Requirements: 10.1, 10.2, 10.3_

  - [ ] 28.3 Test alert threshold configuration
    - Test modifying alert_threshold from 60% to other values
    - Verify AI verification trigger threshold updates correctly
    - Verify alert generation threshold updates correctly
    - Test different thresholds for different camera zones (if applicable)
    - _Requirements: 7.2, 7.3_

  - [ ]* 28.4 Write configuration validation tests
    - Test config file parsing with various formats
    - Test threshold validation logic
    - Test default values when config missing
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 29. Final integration checkpoint and validation report
  - Run all integration tests (tasks 21-28)
  - Run all property-based tests (28 properties from tasks 1-20)
  - Run all unit tests
  - Process all three test videos end-to-end
  - Generate comprehensive test report with:
    - ML vs AI score correlation analysis
    - Performance metrics (latency, throughput)
    - Alert generation statistics
    - Edge case handling results
    - Feature integration validation
  - Verify all requirements (1-11) validated by tests
  - Ensure all tests pass, ask the user if questions arise

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property-based tests validate universal correctness properties (28 total)
- Unit tests validate specific examples, edge cases, and integration points
- Checkpoints ensure incremental validation at key milestones
- The two-tier architecture separates ML pattern detection from AI context verification
- OR logic for alerting ensures fail-safe operation (either system can escalate)
- Both scores are logged separately for forensic analysis and system tuning
- Integration tests (tasks 21-29) validate VLM-ML coordination, performance, and operator workflows
