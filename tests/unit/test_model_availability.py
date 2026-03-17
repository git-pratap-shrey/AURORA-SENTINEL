"""
Unit tests for model availability tracking
Tests circuit breaker pattern implementation
"""
import pytest
import time
import sys
import os

# Add ai-intelligence-layer to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ai-intelligence-layer'))

from model_availability import ModelAvailabilityTracker, ModelStatus


class TestModelAvailabilityTracker:
    """Test suite for ModelAvailabilityTracker"""
    
    def setup_method(self):
        """Create fresh tracker for each test"""
        self.tracker = ModelAvailabilityTracker()
    
    def test_initial_state(self):
        """Test that all models start as available"""
        assert self.tracker.is_available('qwen2vl')
        assert self.tracker.is_available('nemotron')
        assert self.tracker.is_available('ollama')
        
        status = self.tracker.get_status('qwen2vl')
        assert status['available'] is True
        assert status['consecutive_failures'] == 0
        assert status['total_failures'] == 0
        assert status['total_successes'] == 0
    
    def test_record_success(self):
        """Test recording successful model execution"""
        self.tracker.record_success('qwen2vl')
        
        status = self.tracker.get_status('qwen2vl')
        assert status['available'] is True
        assert status['consecutive_failures'] == 0
        assert status['total_successes'] == 1
        assert status['last_success_time'] is not None
    
    def test_record_single_failure(self):
        """Test recording a single failure doesn't mark unavailable"""
        self.tracker.record_failure('qwen2vl')
        
        status = self.tracker.get_status('qwen2vl')
        assert status['available'] is True  # Still available after 1 failure
        assert status['consecutive_failures'] == 1
        assert status['total_failures'] == 1
        assert status['last_failure_time'] is not None
    
    def test_record_two_failures(self):
        """Test recording two failures doesn't mark unavailable"""
        self.tracker.record_failure('qwen2vl')
        self.tracker.record_failure('qwen2vl')
        
        status = self.tracker.get_status('qwen2vl')
        assert status['available'] is True  # Still available after 2 failures
        assert status['consecutive_failures'] == 2
        assert status['total_failures'] == 2
    
    def test_three_consecutive_failures_marks_unavailable(self):
        """Test that 3 consecutive failures marks model unavailable"""
        self.tracker.record_failure('qwen2vl')
        self.tracker.record_failure('qwen2vl')
        self.tracker.record_failure('qwen2vl')
        
        status = self.tracker.get_status('qwen2vl')
        assert status['available'] is False  # Unavailable after 3 failures
        assert status['consecutive_failures'] == 3
        assert status['total_failures'] == 3
        assert not self.tracker.is_available('qwen2vl')
    
    def test_success_resets_consecutive_failures(self):
        """Test that success resets consecutive failure count"""
        self.tracker.record_failure('qwen2vl')
        self.tracker.record_failure('qwen2vl')
        self.tracker.record_success('qwen2vl')
        
        status = self.tracker.get_status('qwen2vl')
        assert status['available'] is True
        assert status['consecutive_failures'] == 0
        assert status['total_failures'] == 2  # Total still tracked
        assert status['total_successes'] == 1
    
    def test_cooldown_period_blocks_retry(self):
        """Test that unavailable model is blocked during cooldown"""
        # Mark unavailable
        self.tracker.record_failure('qwen2vl')
        self.tracker.record_failure('qwen2vl')
        self.tracker.record_failure('qwen2vl')
        
        assert not self.tracker.is_available('qwen2vl')
        
        # Should still be unavailable immediately
        assert not self.tracker.is_available('qwen2vl')
    
    def test_cooldown_period_allows_retry_after_timeout(self):
        """Test that model becomes available after cooldown period"""
        # Temporarily reduce cooldown for testing
        original_cooldown = self.tracker.COOLDOWN_SECONDS
        self.tracker.COOLDOWN_SECONDS = 1  # 1 second for testing
        
        try:
            # Mark unavailable
            self.tracker.record_failure('qwen2vl')
            self.tracker.record_failure('qwen2vl')
            self.tracker.record_failure('qwen2vl')
            
            assert not self.tracker.is_available('qwen2vl')
            
            # Wait for cooldown
            time.sleep(1.1)
            
            # Should be available again
            assert self.tracker.is_available('qwen2vl')
            
            # Consecutive failures should be reset
            status = self.tracker.get_status('qwen2vl')
            assert status['consecutive_failures'] == 0
        finally:
            self.tracker.COOLDOWN_SECONDS = original_cooldown
    
    def test_independent_model_tracking(self):
        """Test that different models are tracked independently"""
        # Fail qwen2vl
        self.tracker.record_failure('qwen2vl')
        self.tracker.record_failure('qwen2vl')
        self.tracker.record_failure('qwen2vl')
        
        # Succeed with ollama
        self.tracker.record_success('ollama')
        
        # Check states
        assert not self.tracker.is_available('qwen2vl')
        assert self.tracker.is_available('ollama')
        assert self.tracker.is_available('nemotron')
        
        qwen_status = self.tracker.get_status('qwen2vl')
        ollama_status = self.tracker.get_status('ollama')
        
        assert qwen_status['consecutive_failures'] == 3
        assert ollama_status['consecutive_failures'] == 0
        assert ollama_status['total_successes'] == 1
    
    def test_get_all_status(self):
        """Test getting status for all models"""
        self.tracker.record_success('qwen2vl')
        self.tracker.record_failure('ollama')
        
        all_status = self.tracker.get_all_status()
        
        assert 'qwen2vl' in all_status
        assert 'nemotron' in all_status
        assert 'ollama' in all_status
        
        assert all_status['qwen2vl']['total_successes'] == 1
        assert all_status['ollama']['total_failures'] == 1
    
    def test_reset_single_model(self):
        """Test resetting a single model's status"""
        self.tracker.record_failure('qwen2vl')
        self.tracker.record_failure('qwen2vl')
        self.tracker.record_failure('qwen2vl')
        
        assert not self.tracker.is_available('qwen2vl')
        
        self.tracker.reset('qwen2vl')
        
        status = self.tracker.get_status('qwen2vl')
        assert status['available'] is True
        assert status['consecutive_failures'] == 0
        assert status['total_failures'] == 0
        assert self.tracker.is_available('qwen2vl')
    
    def test_reset_all_models(self):
        """Test resetting all models"""
        self.tracker.record_failure('qwen2vl')
        self.tracker.record_failure('ollama')
        self.tracker.record_success('nemotron')
        
        self.tracker.reset()
        
        for model in ['qwen2vl', 'nemotron', 'ollama']:
            status = self.tracker.get_status(model)
            assert status['available'] is True
            assert status['consecutive_failures'] == 0
            assert status['total_failures'] == 0
            assert status['total_successes'] == 0
    
    def test_unknown_model(self):
        """Test handling of unknown model names"""
        # Should not crash, just log warning
        self.tracker.record_success('unknown_model')
        self.tracker.record_failure('unknown_model')
        
        assert self.tracker.get_status('unknown_model') is None
        assert not self.tracker.is_available('unknown_model')
    
    def test_failure_with_exception(self):
        """Test recording failure with exception details"""
        error = ValueError("Model initialization failed")
        self.tracker.record_failure('qwen2vl', error)
        
        status = self.tracker.get_status('qwen2vl')
        assert status['consecutive_failures'] == 1
        assert status['total_failures'] == 1
    
    def test_cooldown_remaining_calculation(self):
        """Test that cooldown remaining is calculated correctly"""
        # Temporarily reduce cooldown for testing
        original_cooldown = self.tracker.COOLDOWN_SECONDS
        self.tracker.COOLDOWN_SECONDS = 10  # 10 seconds for testing
        
        try:
            # Mark unavailable
            self.tracker.record_failure('qwen2vl')
            self.tracker.record_failure('qwen2vl')
            self.tracker.record_failure('qwen2vl')
            
            status = self.tracker.get_status('qwen2vl')
            assert status['cooldown_remaining_seconds'] is not None
            assert status['cooldown_remaining_seconds'] > 0
            assert status['cooldown_remaining_seconds'] <= 10
        finally:
            self.tracker.COOLDOWN_SECONDS = original_cooldown
    
    def test_status_includes_timestamps(self):
        """Test that status includes ISO formatted timestamps"""
        self.tracker.record_failure('qwen2vl')
        time.sleep(0.1)
        self.tracker.record_success('qwen2vl')
        
        status = self.tracker.get_status('qwen2vl')
        
        assert status['last_failure_time'] is not None
        assert status['last_success_time'] is not None
        assert 'T' in status['last_failure_time']  # ISO format
        assert 'T' in status['last_success_time']  # ISO format
        
        assert status['time_since_failure_seconds'] is not None
        assert status['time_since_success_seconds'] is not None
        assert status['time_since_success_seconds'] < status['time_since_failure_seconds']


class TestModelAvailabilityIntegration:
    """Integration tests for model availability tracking"""
    
    def test_realistic_failure_recovery_scenario(self):
        """Test a realistic scenario of failures and recovery"""
        tracker = ModelAvailabilityTracker()
        tracker.COOLDOWN_SECONDS = 1  # Short cooldown for testing
        
        # Simulate intermittent failures
        tracker.record_success('qwen2vl')
        tracker.record_success('qwen2vl')
        tracker.record_failure('qwen2vl')
        tracker.record_success('qwen2vl')  # Recovers
        
        assert tracker.is_available('qwen2vl')
        
        # Now simulate sustained failures
        tracker.record_failure('qwen2vl')
        tracker.record_failure('qwen2vl')
        tracker.record_failure('qwen2vl')
        
        assert not tracker.is_available('qwen2vl')
        
        # Wait for cooldown
        time.sleep(1.1)
        
        # Should be available for retry
        assert tracker.is_available('qwen2vl')
        
        # Successful retry
        tracker.record_success('qwen2vl')
        
        status = tracker.get_status('qwen2vl')
        assert status['available'] is True
        assert status['consecutive_failures'] == 0
        assert status['total_successes'] == 4
        assert status['total_failures'] == 4
    
    def test_multiple_models_failing_independently(self):
        """Test multiple models failing at different rates"""
        tracker = ModelAvailabilityTracker()
        
        # Qwen2VL fails quickly
        tracker.record_failure('qwen2vl')
        tracker.record_failure('qwen2vl')
        tracker.record_failure('qwen2vl')
        
        # Ollama has intermittent issues
        tracker.record_success('ollama')
        tracker.record_failure('ollama')
        tracker.record_success('ollama')
        
        # Nemotron works fine
        tracker.record_success('nemotron')
        tracker.record_success('nemotron')
        
        # Check final states
        assert not tracker.is_available('qwen2vl')
        assert tracker.is_available('ollama')
        assert tracker.is_available('nemotron')
        
        all_status = tracker.get_all_status()
        assert all_status['qwen2vl']['available'] is False
        assert all_status['ollama']['available'] is True
        assert all_status['nemotron']['available'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
