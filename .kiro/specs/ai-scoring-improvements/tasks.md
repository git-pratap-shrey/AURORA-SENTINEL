# Implementation Plan: AI Scoring Improvements

## Overview

This implementation plan addresses three critical issues in the AI threat detection system: (1) eliminating hardcoded fallback scores, (2) implementing strict threat classification without "prank"/"drama" categories, and (3) integrating Nemotron ColEmbed V2 as an embedding-based verification layer. The implementation follows a layered approach: first fixing the keyword-based fallback logic, then updating prompts and classification rules, then integrating Nemotron verification, and finally updating the weighted scoring service.

## Tasks

- [ ] 1. Implement keyword-based fallback scoring in AI router
  - [x] 1.1 Update parse_ai_response function in aiRouter_enhanced.py
    - Remove hardcoded score of 40
    - Implement keyword analysis with score ranges: fight keywords (75-90), sport indicators (20-35), suspicious (60-75), normal (10-25)
    - Add logic to exclude fight classification when sport indicators present
    - Return structured dict with aiScore, sceneType, explanation, confidence
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.7_
  
  - [x] 1.2 Write unit tests for keyword-based fallback
    - Test fight keywords without sport indicators → 75-90 range
    - Test sport indicators → 20-35 range
    - Test suspicious keywords → 60-75 range
    - Test normal keywords → 10-25 range
    - Test mixed keywords (fight + sport) → sport takes precedence
    - Verify no hardcoded 40 or ml_score * 0.6 multipliers
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 2. Update Qwen2-VL integration with strict classification
  - [x] 2.1 Update _parse_response in qwen2vl_integration.py
    - Implement keyword-based fallback matching aiRouter_enhanced.py logic
    - Map keywords to score ranges (fight: 75-90, sport: 20-35, suspicious: 60-75, normal: 10-25)
    - Handle heavy fighting keywords (multiple strikes, injury) → 80-95
    - Return structured dict with aiScore, sceneType, explanation, confidence, parsing_method
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.3, 2.6_
  
  - [x] 2.2 Update analysis prompts to remove prank/drama classifications
    - Update STRICT_THREAT_PROMPT to only allow: real_fight, organized_sport, suspicious, normal
    - Add explicit instruction: "DO NOT classify as 'prank' or 'drama'"
    - Add sport indicators: protective gear, referee, ring structure
    - Add heavy fighting indicators: multiple strikes, sustained aggression, visible injury
    - Update VIDEO_ANALYSIS_PROMPT with same strict classification rules
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.8, 5.1, 5.2, 5.3, 5.4, 5.7, 5.8_
  
  - [x] 2.3 Write unit tests for strict classification
    - Test that prank/drama are never returned as sceneType
    - Test fight without sport indicators → real_fight
    - Test fight with protective gear + referee → organized_sport (capped at 35)
    - Test heavy fighting keywords → 80-95 range
    - Test suspicious crowd behavior → suspicious (60-75)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

- [x] 3. Checkpoint - Verify keyword fallback and strict classification
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Implement Nemotron embedding verification layer
  - [x] 4.1 Create NemotronProvider class in vlm_service.py
    - Load nvidia/nemotron-colembed-vl-4b-v2 model with AutoModel and AutoProcessor
    - Implement forward_images() method to compute image embeddings
    - Implement forward_queries() method to compute text embeddings
    - Implement get_scores() method to compute cosine similarity between embeddings
    - Add error handling for model loading failures
    - _Requirements: 3.1, 3.2, 3.13, 6.2, 6.3_
  
  - [x] 4.2 Implement verify_analysis method in NemotronProvider
    - Compute image embedding using forward_images()
    - Compute Qwen summary embedding using forward_queries()
    - Define predefined threat category queries: real_fight, organized_sport, normal, suspicious
    - Compute category embeddings using forward_queries()
    - Calculate verification score (image ↔ Qwen summary similarity)
    - Calculate category scores (image ↔ threat category similarities)
    - Determine Nemotron's scene classification (highest category score)
    - Map category similarity to risk scores: real_fight (80-95), organized_sport (20-35), normal (10-25), suspicious (60-75)
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.10_
  
  - [x] 4.3 Implement agreement logic and score recommendation
    - Mark as "verified" if verification score > 0.6
    - Check if Nemotron scene type matches Qwen scene type
    - If verified AND agreement: use average of both scores, confidence 0.9
    - If mismatch OR disagreement: use higher risk score (conservative), confidence 0.5
    - Return complete verification result with all scores and metadata
    - Add timeout handling (3 second limit)
    - _Requirements: 3.5, 3.6, 3.8, 3.9, 3.11, 3.12, 3.13, 7.2, 7.3_
  
  - [x] 4.4 Write property test for Nemotron verification
    - **Property 1: Conservative disagreement handling**
    - **Validates: Requirements 3.9**
    - Test that when models disagree, the higher risk score is always used
    - Test that verification score < 0.6 triggers conservative handling
    - _Requirements: 3.9_
    - _Note: Property validated in test_conservative_disagreement_property method_
  
  - [x] 4.5 Write unit tests for Nemotron verification
    - Test verified agreement case (verification > 0.6, same scene type) → average score
    - Test mismatch case (verification < 0.6) → higher risk score
    - Test disagreement case (different scene types) → higher risk score
    - Test category score mapping to risk ranges
    - Test timeout handling (> 3 seconds)
    - Test model unavailable fallback
    - _Requirements: 3.5, 3.6, 3.8, 3.9, 3.10, 3.11, 3.13_

- [ ] 5. Integrate Nemotron into AI analysis pipeline
  - [x] 5.1 Update analyze_image in aiRouter_enhanced.py to call Nemotron
    - After Qwen2-VL analysis, extract text summary and scene type
    - Call NemotronProvider.verify_analysis() with image, summary, scene type
    - Handle Nemotron unavailable case (use Qwen score, log fallback)
    - Include Nemotron verification result in final response
    - Add latency tracking for Nemotron calls
    - _Requirements: 3.1, 3.11, 3.12, 6.3, 7.2, 7.5_
  
  - [x] 5.2 Write integration tests for Nemotron pipeline
    - Test full pipeline: Qwen → Nemotron → final score
    - Test Nemotron unavailable fallback
    - Test timeout handling
    - Test that Nemotron verification details are included in response
    - _Requirements: 3.11, 3.12, 6.3_

- [ ] 6. Update weighted scoring service
  - [x] 6.1 Update calculate_scores in scoring_service.py
    - Remove hardcoded ml_score * 0.6 fallback
    - Implement proper fallback: AI unavailable → Final = ML, confidence 0.3
    - Implement proper fallback: ML unavailable → Final = AI, confidence 0.6
    - Use Nemotron-adjusted AI score in weighted calculation (0.3 * ML + 0.7 * AI)
    - Add scoring_method metadata: 'weighted', 'ml_only', 'ai_only'
    - Include Nemotron verification details in response
    - Log all component scores (ML, AI, Nemotron adjustment) for audit
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_
  
  - [x] 6.2 Write property test for weighted scoring
    - **Property 2: Score consistency**
    - **Validates: Requirements 4.1, 4.2, 4.3**
    - Test that Final = 0.3 * ML + 0.7 * AI when both available
    - Test that Final = ML when AI unavailable
    - Test that Final = AI when ML unavailable
    - Test that no arbitrary multipliers are applied
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  
  - [x] 6.3 Write unit tests for scoring service
    - Test weighted calculation (both scores available)
    - Test ML-only fallback (AI unavailable)
    - Test AI-only fallback (ML unavailable)
    - Test Nemotron-adjusted AI score in weighted calculation
    - Test metadata includes scoring_method
    - Test component scores logged for audit
    - Verify no hardcoded 40 or ml_score * 0.6
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [x] 7. Checkpoint - Verify end-to-end scoring pipeline
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement robust error handling
  - [x] 8.1 Add model availability tracking
    - Track consecutive failures for each model (Qwen2-VL, Nemotron, Ollama)
    - Mark model unavailable after 3 consecutive failures
    - Skip unavailable models for 5 minutes before retrying
    - Report model status in health check endpoints
    - _Requirements: 6.6, 6.7_
  
  - [x] 8.2 Add comprehensive error handling and logging
    - Handle Qwen2-VL load failure → attempt Ollama fallback
    - Handle Nemotron load failure → log and continue with single-model analysis
    - Handle Nemotron timeout → use Qwen score and log timeout
    - Handle both Qwen and Ollama failure → use ML score as final
    - Ensure all error paths return valid score (never null/undefined)
    - Include error details in analysis response
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.8_
  
  - [x] 8.3 Write unit tests for error handling
    - Test Qwen failure → Ollama fallback
    - Test Nemotron failure → single-model analysis
    - Test both AI models fail → ML score used
    - Test consecutive failure tracking
    - Test model unavailable skip logic (5 minute cooldown)
    - Test that all error paths return valid scores
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.7, 6.8_

- [ ] 9. Add performance optimizations
  - [x] 9.1 Implement ML score threshold optimization
    - Skip AI analysis when ML score < 20 (low threat, conserve resources)
    - Log skipped AI analysis for monitoring
    - _Requirements: 7.6_
  
  - [x] 9.2 Add latency tracking and timeout enforcement
    - Track Qwen2-VL latency (target: 2 seconds on GPU)
    - Track Nemotron latency (target: 3 seconds)
    - Enforce 5 second total timeout for combined analysis
    - Log latency metrics in response metadata
    - Timeout and use partial results if limits exceeded
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [x] 9.3 Write performance tests
    - Test Qwen2-VL completes within 2 seconds on GPU
    - Test Nemotron completes within 3 seconds
    - Test combined analysis completes within 5 seconds
    - Test timeout handling returns partial results
    - Test ML threshold skip (ML < 20 → no AI analysis)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.6_

- [x] 10. Final integration and validation
  - [x] 10.1 Test with real video samples
    - Test fight videos → expect scores 75-95, scene type real_fight
    - Test boxing videos → expect scores 20-35 (capped), scene type organized_sport
    - Test normal videos → expect scores 10-25, scene type normal
    - Test suspicious behavior → expect scores 60-75, scene type suspicious
    - Verify no scores cluster around 40
    - Verify no prank/drama classifications appear
    - _Requirements: 1.2, 1.3, 1.4, 2.3, 2.4, 2.5, 2.6, 2.7_
  
  - [x] 10.2 Test Nemotron verification scenarios
    - Test agreement case (both models agree) → average score used
    - Test disagreement case (models disagree) → higher risk score used
    - Test mismatch case (verification < 0.6) → conservative handling
    - Test Nemotron unavailable → Qwen score used
    - Verify verification details included in response
    - _Requirements: 3.8, 3.9, 3.11, 3.12_
  
  - [x] 10.3 Validate fallback scenarios
    - Test JSON parse failure → keyword analysis used
    - Test Qwen failure → Ollama fallback
    - Test both AI models fail → ML score used
    - Test Nemotron timeout → Qwen score used
    - Verify no hardcoded 40 or ml_score * 0.6 in any scenario
    - _Requirements: 1.1, 1.2, 1.5, 1.6, 4.2, 4.3, 6.1, 6.3, 6.4_

- [x] 11. Final checkpoint - Complete system validation
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Nemotron uses embedding-based verification (NOT generative), using forward_queries(), forward_images(), and get_scores()
- Conservative approach: when models disagree, use higher risk score
- No hardcoded scores (40 or ml_score * 0.6) anywhere in the system
- Strict classification: only real_fight, organized_sport, normal, suspicious (no prank/drama)
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
