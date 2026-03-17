# Requirements Document: Enhanced Fight Detection

## Introduction

The Aurora Sentinel surveillance system currently fails to detect actual fights in video footage, producing risk scores below 30% for complete fight sequences. This feature will improve fight detection accuracy by making the ML layer aggressive and sensitive to ANY fighting-like behavior (punches, strikes, grappling, proximity violations), while delegating the smart discrimination (boxing vs real fight, drama vs violence) to the AI intelligence layer. The system will use a two-tier approach: ML detects physical combat patterns aggressively, AI verifies context and intent, and if EITHER system escalates, the operator is alerted for manual review.

## Glossary

- **Risk_Engine**: The component responsible for calculating threat scores from detection data
- **Aggression_Detector**: The subsystem that analyzes pose keypoints to identify fighting stances and strikes
- **Temporal_Validator**: The component that validates risk scores across multiple frames to reduce false positives
- **Proximity_Analyzer**: The subsystem that detects when people are close enough for physical confrontation
- **Risk_Score**: A percentage value (0-100) representing the threat level of detected activity
- **ML_Score**: The risk score produced by the machine learning detection engine
- **AI_Score**: The risk score produced by the AI intelligence verification layer
- **Final_Score**: The maximum of ML_Score and AI_Score, used for operator alerts
- **Fighting_Pose**: Body posture characterized by raised arms, widened stance, or extended limbs indicating combat
- **Sustained_Threat**: A high-risk situation that persists across multiple consecutive frames
- **Strike_Motion**: Rapid limb extension detected through pose keypoint velocity analysis
- **Close_Combat**: Physical interaction where person-to-person distance is less than 40% of average body height

## Requirements

### Requirement 1: Temporal Validation Adjustment

**User Story:** As a security operator, I want the system to detect sustained fights without requiring 50% of frames to be high-risk, so that real violent incidents generate alerts even when brief pauses occur during the confrontation.

#### Acceptance Criteria

1. WHEN analyzing a video sequence, THE Temporal_Validator SHALL require at most 30% of frames in the validation window to exceed the risk threshold for critical threats
2. WHEN a fight contains brief pauses or repositioning, THE Temporal_Validator SHALL maintain elevated risk scores across the sequence
3. WHEN the validation window contains 20 or more frames, THE Temporal_Validator SHALL apply temporal validation rules
4. THE Temporal_Validator SHALL use a 20-frame minimum window instead of 30 frames for faster threat detection
5. WHEN temporal validation reduces a risk score, THE Risk_Engine SHALL apply at most a 0.4 suppression multiplier instead of 0.6

### Requirement 2: Aggression Detection Threshold Calibration

**User Story:** As a security operator, I want fighting poses and strikes to accumulate higher aggression scores, so that ANY combat-like behavior is properly identified as high-risk by the ML layer.

#### Acceptance Criteria

1. WHEN a person has both arms raised above shoulders with a widened stance, THE Aggression_Detector SHALL assign an aggression score of at least 0.7
2. WHEN a person extends an arm rapidly (strike motion), THE Aggression_Detector SHALL assign an aggression score of at least 0.5
3. WHEN a person has hands near head with feet spread wider than 30% of body height, THE Aggression_Detector SHALL assign an aggression score of at least 0.6
4. WHEN multiple aggressive indicators are present simultaneously, THE Aggression_Detector SHALL accumulate scores up to a maximum of 1.0
5. THE Aggression_Detector SHALL NOT reduce scores for controlled or rhythmic movements - all combat poses SHALL be scored equally

### Requirement 3: Proximity Violation Sensitivity

**User Story:** As a security operator, I want close combat situations to be detected when people are within striking distance, so that physical altercations trigger appropriate risk escalation.

#### Acceptance Criteria

1. WHEN two people are within 40% of their average body height, THE Proximity_Analyzer SHALL register a proximity violation
2. WHEN a proximity violation occurs AND either person shows aggression score above 0.3, THE Proximity_Analyzer SHALL escalate the violation weight by 3.0
3. WHEN multiple proximity violations occur in a single frame, THE Proximity_Analyzer SHALL return a score proportional to the violation count
4. WHEN people are close but both show aggression scores below 0.3, THE Proximity_Analyzer SHALL apply a baseline violation weight of 1.5
5. THE Proximity_Analyzer SHALL calculate distance using person-to-person center points normalized by average body height

### Requirement 4: Risk Score Escalation for Combat

**User Story:** As a security operator, I want the risk score to escalate quickly when fighting is detected, so that alerts are generated before the situation becomes critical.

#### Acceptance Criteria

1. WHEN aggression score exceeds 0.6 AND proximity violation is detected, THE Risk_Engine SHALL produce a minimum risk score of 70%
2. WHEN aggression score exceeds 0.5 for any person, THE Risk_Engine SHALL apply a suppression factor of 1.0 to prevent dampening
3. WHEN grappling or clinching is detected (distance < 40% of height), THE Risk_Engine SHALL produce a minimum risk score of 65%
4. WHEN strike motion is detected with proximity violation, THE Risk_Engine SHALL escalate the raw score by at least 0.3
5. WHEN multiple high-risk factors are present, THE Risk_Engine SHALL apply multiplicative escalation rather than simple addition

### Requirement 5: Strike Velocity Detection

**User Story:** As a security operator, I want rapid limb movements to be detected as potential strikes, so that punching and kicking actions contribute to fight detection.

#### Acceptance Criteria

1. WHEN a person's wrist moves more than 40% of body height between consecutive frames, THE Aggression_Detector SHALL register a strike motion
2. WHEN strike motion is detected AND the person has an extended arm, THE Aggression_Detector SHALL add 0.4 to the aggression score
3. WHEN analyzing tracked persons, THE Aggression_Detector SHALL maintain per-keypoint velocity history for wrist and ankle positions
4. WHEN strike velocity is detected in combination with proximity violation, THE Risk_Engine SHALL escalate the risk score by 0.3
5. THE Aggression_Detector SHALL normalize velocity thresholds by person body height to handle different camera distances

### Requirement 6: Multi-Signal Validation Adjustment

**User Story:** As a security operator, I want the system to trust high aggression signals without requiring multiple corroborating factors, so that clear fighting behavior generates alerts even when other factors are absent.

#### Acceptance Criteria

1. WHEN aggression score exceeds 0.6, THE Risk_Engine SHALL set suppression factor to 1.0 regardless of other signal counts
2. WHEN fewer than 2 high-risk signals are present BUT aggression exceeds 0.5, THE Risk_Engine SHALL apply at most 0.8 suppression instead of 0.6
3. WHEN weapon detection exceeds 0.4, THE Risk_Engine SHALL bypass multi-signal validation entirely
4. THE Risk_Engine SHALL count proximity violations as high-risk signals when combined with aggression above 0.3
5. WHEN grappling is detected, THE Risk_Engine SHALL bypass multi-signal validation requirements

### Requirement 7: ML-AI Two-Tier Detection Architecture

**User Story:** As a security operator, I want the ML to aggressively detect any fighting-like behavior and the AI to provide context-aware verification, so that I receive alerts when EITHER system escalates, ensuring no real threats are missed.

#### Acceptance Criteria

1. THE Risk_Engine SHALL NOT attempt to distinguish between boxing, sparring, drama, or real fights - it SHALL detect all combat-like physical patterns
2. WHEN the ML Risk_Engine produces a score above 60%, THE system SHALL trigger AI intelligence verification in parallel
3. WHEN the AI intelligence layer produces a risk score above 60%, THE system SHALL generate an alert regardless of ML score
4. WHEN EITHER ML score OR AI score exceeds 60%, THE system SHALL flag the incident as requiring operator review
5. THE final displayed risk score SHALL be the MAXIMUM of (ML score, AI score) to ensure operator visibility of any escalation
6. THE system SHALL log both ML and AI scores separately for forensic analysis and system tuning
7. THE AI intelligence layer SHALL receive the ML score and detection factors as context for its analysis

### Requirement 8: Grappling and Clinch Detection

**User Story:** As a security operator, I want grappling and clinching to be detected as high-risk combat, so that ground fights and wrestling-style altercations generate appropriate alerts.

#### Acceptance Criteria

1. WHEN two people are within 40% of average body height AND bounding boxes overlap by more than 60%, THE Risk_Engine SHALL detect grappling
2. WHEN grappling is detected, THE Risk_Engine SHALL assign a grappling factor score of 0.8
3. WHEN grappling is detected, THE Risk_Engine SHALL produce a minimum risk score of 65%
4. THE Risk_Engine SHALL set suppression factor to 1.0 when grappling is detected
5. WHEN grappling persists for more than 10 frames, THE Risk_Engine SHALL maintain elevated risk scores throughout the sequence

### Requirement 9: Test Video Validation

**User Story:** As a security operator, I want the enhanced system to correctly detect combat in the test videos, so that I can verify the improvements are working as intended.

#### Acceptance Criteria

1. WHEN processing storage/temp/raw_1772268166.802879_4rth.mp4, THE ML Risk_Engine SHALL produce a maximum risk score above 70%
2. WHEN processing storage/temp/raw_1772268210.415873_5th.mp4, THE ML Risk_Engine SHALL produce a maximum risk score above 70%
3. WHEN processing storage/temp/raw_1772268625.365338_scr9-231238~2sdddsd.mp4, THE ML Risk_Engine SHALL produce a maximum risk score above 60% (boxing has fighting poses)
4. WHEN processing the boxing video, THE AI intelligence layer SHALL identify it as "controlled sparring" or "boxing" and MAY reduce its AI score below 30%
5. THE final system score (max of ML and AI) SHALL trigger operator alerts for all three videos, allowing manual verification
6. WHEN processing fight videos, THE system SHALL generate alerts with level 'critical' or 'high' based on the maximum score

### Requirement 10: Calibration and Threshold Documentation

**User Story:** As a developer, I want all threshold values and calibration parameters to be documented and configurable, so that future adjustments can be made systematically.

#### Acceptance Criteria

1. THE Risk_Engine SHALL expose configurable parameters for temporal validation thresholds
2. THE Aggression_Detector SHALL expose configurable parameters for pose-based scoring weights
3. THE Proximity_Analyzer SHALL expose configurable parameters for distance thresholds
4. THE Risk_Engine SHALL log threshold values and detection parameters when processing video
5. WHEN threshold parameters are modified, THE Risk_Engine SHALL validate that values are within acceptable ranges (0.0 to 1.0 for scores, positive values for distances)

### Requirement 11: Operator Alert Escalation Logic

**User Story:** As a security operator, I want to be alerted whenever EITHER the ML or AI system detects high risk, so that I can manually verify any potential threat without relying on perfect automated discrimination.

#### Acceptance Criteria

1. WHEN ML_Score exceeds 60% OR AI_Score exceeds 60%, THE system SHALL generate an operator alert
2. THE alert SHALL display both ML_Score and AI_Score separately for operator context
3. THE alert SHALL use the Final_Score (maximum of ML and AI) for priority ranking and visual indicators
4. WHEN ML_Score is high but AI_Score is low, THE alert SHALL indicate "ML Detection - AI Verification Pending"
5. WHEN AI_Score is high but ML_Score is low, THE alert SHALL indicate "AI Detection - Review ML Factors"
6. THE system SHALL color-code alerts: RED when Final_Score > 70%, ORANGE when Final_Score > 50%, YELLOW when Final_Score > 30%
7. THE operator interface SHALL allow filtering alerts by ML_Score, AI_Score, or Final_Score thresholds
