# Design Document: Enhanced Fight Detection

## Overview

The Aurora Sentinel surveillance system currently fails to detect actual fights in video footage, producing risk scores below 30% for complete fight sequences. This design implements a two-tier detection architecture where the ML layer aggressively detects ANY combat-like behavior (punches, strikes, grappling, proximity violations) without attempting to discriminate between boxing, sparring, drama, or real fights. The AI intelligence layer then provides context-aware verification to distinguish controlled activities from genuine threats.

The system uses a dual-scoring approach:
- **ML_Score**: Aggressive detection of physical combat patterns (no smart discrimination)
- **AI_Score**: Context-aware verification using vision-language models
- **Final_Score**: MAX(ML_Score, AI_Score) for operator alerts

If EITHER score exceeds the alert threshold (60%), the operator is notified for manual review. This ensures no real threats are missed while providing context for false positive reduction.

### Key Design Principles

1. **Separation of Concerns**: ML detects patterns, AI interprets context
2. **Fail-Safe Alerting**: OR logic ensures any escalation reaches operators
3. **Forensic Transparency**: Both scores logged for system tuning
4. **Aggressive ML Detection**: Remove all discrimination logic from pose analysis
5. **Parallel Processing**: ML and AI run independently to avoid bottlenecks

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      Video Processing Pipeline                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Frame Detection Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ YOLOv8 Pose  │  │ YOLOv8 Object│  │   Tracking   │         │
│  │  Detection   │  │  Detection   │  │  (ByteTrack) │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Two-Tier Risk Scoring                         │
│                                                                   │
│  ┌────────────────────────────┐  ┌──────────────────────────┐  │
│  │     ML Risk Engine         │  │  AI Intelligence Layer   │  │
│  │  (Aggressive Detection)    │  │  (Context Verification)  │  │
│  │                            │  │                          │  │
│  │  • Aggression Detector     │  │  • VLM Analysis          │  │
│  │  • Proximity Analyzer      │  │  • Context Reasoning     │  │
│  │  • Strike Velocity         │  │  • Scene Understanding   │  │
│  │  • Grappling Detection     │  │                          │  │
│  │  • Temporal Validator      │  │  Input: Frame + ML_Score │  │
│  │                            │  │  Output: AI_Score        │  │
│  │  Output: ML_Score          │  │                          │  │
│  └────────────────────────────┘  └──────────────────────────┘  │
│                 │                              │                 │
│                 └──────────────┬───────────────┘                 │
│                                ▼                                 │
│                    ┌────────────────────────┐                    │
│                    │   Score Aggregator     │                    │
│                    │ Final_Score = MAX(ML,AI)│                    │
│                    └────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Alert Generation                            │
│  IF ML_Score > 60% OR AI_Score > 60% THEN generate_alert()     │
│                                                                   │
│  Alert Metadata:                                                 │
│    • ML_Score: X%                                               │
│    • AI_Score: Y%                                               │
│    • Final_Score: MAX(X, Y)                                     │
│    • Detection Source: ML | AI | Both                           │
│    • Color: RED (>70%) | ORANGE (>50%) | YELLOW (>30%)         │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Frame Ingestion**: Video frames enter the detection pipeline
2. **Detection**: YOLOv8 models extract poses, objects, and tracking IDs
3. **ML Risk Calculation**: Risk engine calculates ML_Score using aggressive thresholds
4. **Parallel AI Verification**: If ML_Score > 60%, trigger AI analysis
5. **Score Aggregation**: Calculate Final_Score = MAX(ML_Score, AI_Score)
6. **Alert Decision**: Generate alert if ML_Score > 60% OR AI_Score > 60%
7. **Operator Notification**: Display both scores with context indicators

## Components and Interfaces

### 1. Enhanced Risk Engine (ML Layer)

**File**: `models/scoring/risk_engine.py`

**Class**: `RiskScoringEngine`

**Key Modifications**:

```python
class RiskScoringEngine:
    def __init__(self, fps=30, bypass_calibration=False):
        # Enhanced thresholds for aggressive detection
        self.thresholds = {
            'temporal_validation_ratio': 0.30,  # Reduced from 0.50
            'temporal_window_size': 20,         # Reduced from 30
            'temporal_suppression_max': 0.4,    # Reduced from 0.6
            'proximity_distance': 0.40,         # 40% of avg height
            'proximity_escalation': 3.0,        # Escalation multiplier
            'strike_velocity': 0.40,            # 40% of body height per frame
            'grappling_distance': 0.40,         # 40% of avg height
            'grappling_overlap': 0.60,          # 60% bbox overlap
            'aggression_raised_arms': 0.7,      # Score for raised arms + stance
            'aggression_strike': 0.5,           # Score for strike motion
            'aggression_fighting_stance': 0.6,  # Score for hands near head + wide feet
        }
        
        # Per-person keypoint velocity tracking
        self.keypoint_history = defaultdict(lambda: {
            'wrists': deque(maxlen=10),
            'ankles': deque(maxlen=10)
        })
    
    def calculate_risk(self, detection_data, context=None):
        """
        Calculate ML_Score using aggressive detection.
        Returns: (ml_score, factors_dict)
        """
        # ... existing logic with modifications
        
    def _analyze_aggression(self, poses):
        """
        Aggressive pose detection - NO discrimination between
        boxing, sparring, or real fights. All combat poses scored equally.
        """
        # ... enhanced logic
        
    def _detect_strike_velocity(self, poses):
        """
        Track per-keypoint velocity for wrists and ankles.
        Detect rapid limb movements indicating strikes.
        """
        # ... new implementation
        
    def _detect_grappling(self, poses):
        """
        Detect grappling/clinching using distance + bbox overlap.
        """
        # ... new implementation
```

**Interface**:
- **Input**: `detection_data` (poses, objects, weapons), `context` (timestamp, sensitivity)
- **Output**: `(ml_score: float, factors: dict)`
- **Factors**: weapon_detection, aggressive_posture, proximity_violation, grappling, strike_velocity, loitering, unattended_object, crowd_density

### 2. AI Intelligence Layer

**File**: `ai-intelligence-layer/aiRouter.js`

**Function**: `analyzeImage`

**Enhanced Interface**:

```javascript
const analyzeImage = ai.defineFlow({
    inputSchema: z.object({
        imageData: z.string(),      // base64 frame
        mlScore: z.number(),         // ML_Score from risk engine
        mlFactors: z.object({}),     // Detection factors
        cameraId: z.string(),
        timestamp: z.number(),
        modelOverride: z.string().optional()
    }),
    outputSchema: z.object({
        aiScore: z.number(),         // AI risk score (0-100)
        explanation: z.string(),     // Context reasoning
        sceneType: z.string(),       // "real_fight" | "boxing" | "drama" | "normal"
        confidence: z.number(),      // AI confidence (0-1)
        provider: z.string()
    })
})
```

**Enhanced Prompt**:
```
System: Aurora Sentinel AI Verification Expert.

Context:
- Camera: {cameraId}
- ML Detection Score: {mlScore}%
- ML Factors: {mlFactors}

The ML system detected combat-like behavior at {mlScore}%. Your task is to verify if this is:
1. A real threat (actual fight, assault, violence)
2. Controlled activity (boxing, sparring, martial arts training)
3. Staged/drama (acting, performance, rehearsal)
4. False positive (normal activity misclassified)

Analyze the image and provide:
- AI Risk Score (0-100): Your assessment of actual threat level
- Scene Type: real_fight | boxing | drama | normal
- Explanation: Brief reasoning for your assessment
- Confidence: How certain are you? (0.0-1.0)

Be objective. If ML detected combat poses but this is clearly controlled (boxing ring, protective gear, training environment), your AI score should be LOW. If this appears to be genuine violence, your AI score should be HIGH.
```

### 3. Score Aggregator

**File**: `backend/services/scoring_service.py` (new)

**Class**: `TwoTierScoringService`

```python
class TwoTierScoringService:
    def __init__(self, risk_engine, ai_client):
        self.risk_engine = risk_engine
        self.ai_client = ai_client
        self.alert_threshold = 60.0
        
    async def calculate_scores(self, frame, detection_data, context):
        """
        Calculate both ML and AI scores, return aggregated result.
        """
        # 1. Calculate ML score
        ml_score, ml_factors = self.risk_engine.calculate_risk(
            detection_data, context
        )
        
        # 2. Trigger AI verification if ML score is elevated
        ai_score = 0.0
        ai_explanation = ""
        ai_scene_type = "normal"
        
        if ml_score > self.alert_threshold:
            ai_result = await self.ai_client.analyze_image(
                frame=frame,
                ml_score=ml_score,
                ml_factors=ml_factors,
                camera_id=context.get('camera_id'),
                timestamp=context.get('timestamp')
            )
            ai_score = ai_result['aiScore']
            ai_explanation = ai_result['explanation']
            ai_scene_type = ai_result['sceneType']
        
        # 3. Calculate final score
        final_score = max(ml_score, ai_score)
        
        # 4. Determine detection source
        if ml_score > self.alert_threshold and ai_score > self.alert_threshold:
            source = "both"
        elif ml_score > self.alert_threshold:
            source = "ml"
        elif ai_score > self.alert_threshold:
            source = "ai"
        else:
            source = "none"
        
        return {
            'ml_score': ml_score,
            'ai_score': ai_score,
            'final_score': final_score,
            'ml_factors': ml_factors,
            'ai_explanation': ai_explanation,
            'ai_scene_type': ai_scene_type,
            'detection_source': source,
            'should_alert': final_score > self.alert_threshold
        }
```

### 4. Alert Generator

**File**: `backend/services/alert_service.py` (enhanced)

**Class**: `AlertService`

```python
class AlertService:
    def generate_alert(self, scoring_result, context):
        """
        Generate operator alert with two-tier score metadata.
        """
        final_score = scoring_result['final_score']
        ml_score = scoring_result['ml_score']
        ai_score = scoring_result['ai_score']
        
        # Determine alert level based on Final_Score
        if final_score > 70:
            level = 'critical'
            color = 'red'
        elif final_score > 50:
            level = 'high'
            color = 'orange'
        elif final_score > 30:
            level = 'medium'
            color = 'yellow'
        else:
            level = 'low'
            color = 'green'
        
        # Generate context message
        source = scoring_result['detection_source']
        if source == 'both':
            message = f"Both ML and AI detected threat"
        elif source == 'ml' and ai_score < 30:
            message = f"ML Detection - AI Verification: {scoring_result['ai_scene_type']}"
        elif source == 'ai':
            message = f"AI Detection - Review ML Factors"
        else:
            message = f"Elevated risk detected"
        
        return {
            'level': level,
            'color': color,
            'final_score': final_score,
            'ml_score': ml_score,
            'ai_score': ai_score,
            'detection_source': source,
            'message': message,
            'ml_factors': scoring_result['ml_factors'],
            'ai_explanation': scoring_result['ai_explanation'],
            'ai_scene_type': scoring_result['ai_scene_type'],
            'timestamp': context.get('timestamp'),
            'camera_id': context.get('camera_id')
        }
```

## Data Models

### DetectionData

```python
@dataclass
class DetectionData:
    """Output from YOLOv8 detection models"""
    poses: List[PoseDetection]
    objects: List[ObjectDetection]
    weapons: List[WeaponDetection]
    timestamp: float
    frame_number: int
```

### PoseDetection

```python
@dataclass
class PoseDetection:
    """YOLO pose keypoints with tracking"""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    keypoints: np.ndarray  # (17, 2) - COCO format
    confidence: np.ndarray  # (17,) - per-keypoint confidence
    person_height: float  # bbox height for normalization
```

### ScoringResult

```python
@dataclass
class ScoringResult:
    """Two-tier scoring output"""
    ml_score: float  # 0-100
    ai_score: float  # 0-100
    final_score: float  # MAX(ml_score, ai_score)
    ml_factors: Dict[str, float]
    ai_explanation: str
    ai_scene_type: str  # "real_fight" | "boxing" | "drama" | "normal"
    ai_confidence: float  # 0-1
    detection_source: str  # "ml" | "ai" | "both" | "none"
    should_alert: bool
    timestamp: float
```

### Alert

```python
@dataclass
class Alert:
    """Operator alert with two-tier metadata"""
    alert_id: str
    level: str  # "critical" | "high" | "medium" | "low"
    color: str  # "red" | "orange" | "yellow" | "green"
    final_score: float
    ml_score: float
    ai_score: float
    detection_source: str
    message: str
    ml_factors: Dict[str, float]
    ai_explanation: str
    ai_scene_type: str
    timestamp: float
    camera_id: str
    frame_data: bytes  # Peak frame image
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

Before defining the final properties, I've analyzed the prework to eliminate redundancy:

**Redundancies Identified**:
1. Properties 4.1, 4.3, 8.3 all test minimum risk score thresholds - can be combined into one comprehensive property
2. Properties 4.2, 6.1, 8.4 all test suppression factor = 1.0 under different conditions - can be combined
3. Properties 3.2, 3.4 both test proximity violation weight assignment - can be combined
4. Properties 7.2, 7.3, 7.4, 11.1 all test alert generation logic - can be combined into one comprehensive property
5. Properties 2.1, 2.2, 2.3 test specific aggression scores - these are distinct and should remain separate
6. Properties 5.1, 5.2 test strike detection - can be combined into one property about strike detection and scoring

**Final Property Set**: After consolidation, we have 25 unique properties that provide comprehensive coverage without redundancy.

### Property 1: Temporal Validation Ratio

*For any* video sequence with a validation window of 20+ frames, if at least 30% of frames exceed the risk threshold, then temporal validation should not suppress the risk score below the critical threshold.

**Validates: Requirements 1.1, 1.3**

### Property 2: Temporal Validation Window Size

*For any* video sequence, the temporal validator should use a 20-frame minimum window for validation calculations.

**Validates: Requirements 1.4**

### Property 3: Temporal Suppression Limit

*For any* risk score that undergoes temporal validation suppression, the suppression multiplier should not exceed 0.4 (i.e., score should not be reduced below 40% of original).

**Validates: Requirements 1.5**

### Property 4: Raised Arms Aggression Score

*For any* pose where both arms are raised above shoulders AND feet are spread wider than 30% of body height, the aggression score should be at least 0.7.

**Validates: Requirements 2.1**

### Property 5: Strike Motion Detection and Scoring

*For any* tracked person whose wrist moves more than 40% of body height between consecutive frames, the system should register a strike motion and add at least 0.4 to the aggression score if the arm is extended.

**Validates: Requirements 2.2, 5.1, 5.2**

### Property 6: Fighting Stance Aggression Score

*For any* pose where hands are within 25% of body height from the head AND feet are spread wider than 30% of body height, the aggression score should be at least 0.6.

**Validates: Requirements 2.3**

### Property 7: Aggression Score Accumulation

*For any* pose with multiple aggressive indicators present simultaneously, the total aggression score should accumulate up to a maximum of 1.0.

**Validates: Requirements 2.4**

### Property 8: Proximity Violation Detection

*For any* pair of people whose center-to-center distance is less than 40% of their average body height, the system should register a proximity violation.

**Validates: Requirements 3.1**

### Property 9: Proximity Violation Weight Escalation

*For any* proximity violation where at least one person has an aggression score above 0.3, the violation weight should be 3.0; otherwise, the baseline weight should be 1.5.

**Validates: Requirements 3.2, 3.4**

### Property 10: Multiple Proximity Violations Scoring

*For any* frame with multiple proximity violations, the proximity score should be proportional to the violation count (weighted sum divided by total pairs).

**Validates: Requirements 3.3**

### Property 11: Minimum Risk Score for High-Risk Scenarios

*For any* detection where (aggression > 0.6 AND proximity violation exists) OR (grappling detected), the risk score should be at least 65%.

**Validates: Requirements 4.1, 4.3, 8.3**

### Property 12: Suppression Factor Bypass

*For any* detection where aggression > 0.6 OR weapon detection > 0.4 OR grappling detected, the suppression factor should be set to 1.0 (no suppression).

**Validates: Requirements 4.2, 6.1, 6.3, 6.5, 8.4**

### Property 13: Strike and Proximity Escalation

*For any* detection where strike motion is detected AND proximity violation exists, the raw risk score should be escalated by at least 0.3.

**Validates: Requirements 4.4, 5.4**

### Property 14: Multi-Signal Suppression Adjustment

*For any* detection where fewer than 2 high-risk signals are present BUT aggression exceeds 0.5, the suppression factor should be at most 0.8 (not 0.6).

**Validates: Requirements 6.2**

### Property 15: Proximity as High-Risk Signal

*For any* detection where proximity violation occurs AND aggression > 0.3, the proximity violation should count as a high-risk signal in multi-signal validation.

**Validates: Requirements 6.4**

### Property 16: AI Verification Trigger

*For any* frame where ML_Score exceeds 60%, the system should trigger AI intelligence verification in parallel.

**Validates: Requirements 7.2**

### Property 17: Alert Generation Logic

*For any* scoring result where ML_Score > 60% OR AI_Score > 60%, the system should generate an operator alert.

**Validates: Requirements 7.3, 7.4, 11.1**

### Property 18: Final Score Calculation

*For any* scoring result with ML_Score and AI_Score, the Final_Score should equal MAX(ML_Score, AI_Score).

**Validates: Requirements 7.5**

### Property 19: Score Logging

*For any* processed frame, the system should log both ML_Score and AI_Score separately in the database or log files.

**Validates: Requirements 7.6**

### Property 20: AI Context Passing

*For any* AI verification request, the AI intelligence layer should receive the ML_Score and detection factors as context.

**Validates: Requirements 7.7**

### Property 21: Grappling Detection

*For any* pair of people where distance < 40% of average height AND bounding box overlap > 60%, the system should detect grappling and assign a grappling factor score of 0.8.

**Validates: Requirements 8.1, 8.2**

### Property 22: Grappling Temporal Persistence

*For any* grappling detection that persists for more than 10 consecutive frames, the system should maintain elevated risk scores throughout the sequence.

**Validates: Requirements 8.5**

### Property 23: Alert Level Assignment

*For any* alert with Final_Score, the alert level should be 'critical' or 'high' when Final_Score > 60%.

**Validates: Requirements 9.6**

### Property 24: Parameter Validation

*For any* threshold parameter modification, the system should validate that score values are within 0.0-1.0 and distance values are positive.

**Validates: Requirements 10.5**

### Property 25: Alert Metadata Completeness

*For any* generated alert, the alert object should contain both ML_Score and AI_Score as separate fields.

**Validates: Requirements 11.2**

### Property 26: Alert Priority Ranking

*For any* alert, the priority ranking and visual indicators should be based on Final_Score (not individual ML or AI scores).

**Validates: Requirements 11.3**

### Property 27: Alert Context Messages

*For any* alert where ML_Score > 60% but AI_Score < 30%, the alert message should indicate "ML Detection - AI Verification"; when AI_Score > 60% but ML_Score < 60%, the message should indicate "AI Detection - Review ML Factors".

**Validates: Requirements 11.4, 11.5**

### Property 28: Alert Color Coding

*For any* alert, the color should be RED when Final_Score > 70%, ORANGE when Final_Score > 50%, YELLOW when Final_Score > 30%.

**Validates: Requirements 11.6**

## Error Handling

### ML Risk Engine Errors

1. **Missing Pose Data**: If no poses detected, return 0.0 score with empty factors
2. **Invalid Keypoints**: Skip poses with < 11 valid keypoints or mean confidence < 0.35
3. **Tracking Loss**: Handle missing track_ids gracefully, assume static if no history
4. **Division by Zero**: Validate person height > 0 before normalization
5. **Timestamp Issues**: Use frame count as fallback if timestamp unavailable

### AI Intelligence Layer Errors

1. **API Timeout**: If AI request times out (>5s), return AI_Score = 0 with explanation "AI verification timeout"
2. **Model Unavailable**: Fall back to ML_Score only, log warning
3. **Invalid Response**: If AI returns malformed data, use AI_Score = 0
4. **Rate Limiting**: Implement exponential backoff for API rate limits
5. **Network Errors**: Retry once, then fail gracefully with AI_Score = 0

### Score Aggregation Errors

1. **Missing Scores**: If either score is None, use 0.0 as default
2. **Invalid Score Range**: Clamp scores to 0-100 range
3. **Concurrent Access**: Use thread-safe operations for score calculation
4. **Database Errors**: Log scores to file if database write fails

### Alert Generation Errors

1. **Missing Metadata**: Generate alert with available data, mark incomplete fields
2. **Frame Extraction Failure**: Generate alert without frame image
3. **Notification Failure**: Queue alert for retry, don't block processing pipeline

## Testing Strategy

### Dual Testing Approach

This feature requires both unit tests and property-based tests for comprehensive coverage:

**Unit Tests**: Focus on specific examples, edge cases, and integration points
- Test specific video files (Requirements 9.1-9.5)
- Test configuration parameter exposure (Requirements 10.1-10.4)
- Test alert UI filtering (Requirement 11.7)
- Test error handling scenarios
- Test API integration between components

**Property-Based Tests**: Verify universal properties across all inputs
- Generate random pose data with varying aggression indicators
- Generate random person pairs with varying distances
- Generate random video sequences with varying risk patterns
- Verify scoring logic holds for all valid inputs
- Minimum 100 iterations per property test

### Property-Based Testing Configuration

**Library**: Use `hypothesis` for Python components, `fast-check` for JavaScript AI layer

**Test Configuration**:
```python
from hypothesis import given, settings
import hypothesis.strategies as st

@settings(max_examples=100)
@given(
    poses=st.lists(st.builds(PoseDetection, ...)),
    context=st.builds(dict, ...)
)
def test_property_X(poses, context):
    """
    Feature: enhanced-fight-detection, Property X: [property text]
    """
    # Test implementation
```

**Tag Format**: Each property test must include a comment:
```python
# Feature: enhanced-fight-detection, Property 1: Temporal Validation Ratio
```

### Test Categories

1. **Aggression Detection Tests**
   - Property tests for pose-based scoring (Properties 4, 5, 6, 7)
   - Unit tests for specific fighting poses
   - Edge cases: partial keypoints, low confidence, occluded poses

2. **Proximity Analysis Tests**
   - Property tests for distance calculations (Properties 8, 9, 10)
   - Unit tests for specific person configurations
   - Edge cases: single person, overlapping bboxes, perspective distortion

3. **Temporal Validation Tests**
   - Property tests for validation logic (Properties 1, 2, 3, 22)
   - Unit tests for specific frame sequences
   - Edge cases: short sequences, all high-risk frames, all low-risk frames

4. **Two-Tier Scoring Tests**
   - Property tests for score aggregation (Properties 16, 17, 18, 19, 20)
   - Unit tests for ML/AI integration
   - Edge cases: AI timeout, missing scores, conflicting scores

5. **Alert Generation Tests**
   - Property tests for alert logic (Properties 23, 25, 26, 27, 28)
   - Unit tests for specific alert scenarios
   - Edge cases: boundary scores (59%, 60%, 61%), missing metadata

6. **Strike and Grappling Tests**
   - Property tests for velocity detection (Property 5)
   - Property tests for grappling detection (Properties 21, 22)
   - Unit tests for specific combat scenarios
   - Edge cases: rapid camera movement, tracking loss

7. **Integration Tests**
   - End-to-end tests with real video files (Requirements 9.1-9.5)
   - Test complete pipeline: detection → ML scoring → AI verification → alert
   - Verify both scores logged correctly
   - Verify operator UI displays both scores

### Test Data

**Synthetic Data Generation**:
- Generate random poses with controlled aggression indicators
- Generate random person pairs with controlled distances
- Generate random video sequences with controlled risk patterns

**Real Video Tests**:
- `storage/temp/raw_1772268166.802879_4rth.mp4` - Expected ML_Score > 70%
- `storage/temp/raw_1772268210.415873_5th.mp4` - Expected ML_Score > 70%
- `storage/temp/raw_1772268625.365338_scr9-231238~2sdddsd.mp4` - Expected ML_Score > 60%, AI_Score < 30%

### Performance Testing

- ML scoring should complete in < 50ms per frame
- AI verification should complete in < 2s per frame
- Parallel processing should not block ML pipeline
- System should handle 30 FPS video in real-time

### Regression Testing

- Maintain test suite for existing functionality
- Verify non-combat scenarios don't trigger false positives
- Verify weapon detection still works correctly
- Verify loitering and crowd density detection unchanged
