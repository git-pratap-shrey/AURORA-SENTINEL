"""
Integration test for health check endpoint with model availability status
"""
import pytest
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ai-intelligence-layer'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from model_availability import get_tracker


class TestHealthCheckModelStatus:
    """Test health check integration with model availability tracking"""
    
    def test_get_model_status_returns_all_models(self):
        """Test that get_model_status returns status for all models"""
        from aiRouter_enhanced import get_model_status
        
        status = get_model_status()
        
        # Should have all three models
        assert 'qwen2vl' in status
        assert 'nemotron' in status
        assert 'ollama' in status
        
        # Each should have required fields
        for model_name, model_status in status.items():
            assert 'name' in model_status
            assert 'available' in model_status
            assert 'consecutive_failures' in model_status
            assert 'total_failures' in model_status
            assert 'total_successes' in model_status
    
    def test_model_status_reflects_failures(self):
        """Test that model status reflects recorded failures"""
        from aiRouter_enhanced import get_model_status
        
        tracker = get_tracker()
        tracker.reset()  # Start fresh
        
        # Record some failures
        tracker.record_failure('qwen2vl')
        tracker.record_failure('qwen2vl')
        
        status = get_model_status()
        
        assert status['qwen2vl']['consecutive_failures'] == 2
        assert status['qwen2vl']['total_failures'] == 2
        assert status['qwen2vl']['available'] is True  # Still available after 2
    
    def test_model_status_reflects_unavailability(self):
        """Test that model status shows unavailable after threshold"""
        from aiRouter_enhanced import get_model_status
        
        tracker = get_tracker()
        tracker.reset()  # Start fresh
        
        # Record 3 failures to mark unavailable
        tracker.record_failure('ollama')
        tracker.record_failure('ollama')
        tracker.record_failure('ollama')
        
        status = get_model_status()
        
        assert status['ollama']['consecutive_failures'] == 3
        assert status['ollama']['available'] is False
        assert status['ollama']['cooldown_remaining_seconds'] is not None
    
    def test_model_status_reflects_successes(self):
        """Test that model status reflects successful operations"""
        from aiRouter_enhanced import get_model_status
        
        tracker = get_tracker()
        tracker.reset()  # Start fresh
        
        # Record successes
        tracker.record_success('nemotron')
        tracker.record_success('nemotron')
        tracker.record_success('nemotron')
        
        status = get_model_status()
        
        assert status['nemotron']['total_successes'] == 3
        assert status['nemotron']['consecutive_failures'] == 0
        assert status['nemotron']['available'] is True
        assert status['nemotron']['last_success_time'] is not None
    
    def test_health_check_format(self):
        """Test that health check returns expected format"""
        from aiRouter_enhanced import get_model_status
        
        tracker = get_tracker()
        tracker.reset()
        
        # Simulate mixed states
        tracker.record_success('qwen2vl')
        tracker.record_failure('ollama')
        tracker.record_failure('ollama')
        
        status = get_model_status()
        
        # Verify structure matches what health endpoint expects
        assert isinstance(status, dict)
        assert len(status) == 3
        
        for model_name in ['qwen2vl', 'nemotron', 'ollama']:
            assert model_name in status
            model_info = status[model_name]
            
            # Required fields for health check
            assert isinstance(model_info['name'], str)
            assert isinstance(model_info['available'], bool)
            assert isinstance(model_info['consecutive_failures'], int)
            assert isinstance(model_info['total_failures'], int)
            assert isinstance(model_info['total_successes'], int)


class TestModelAvailabilityInAnalysis:
    """Test that model availability affects analysis flow"""
    
    def test_unavailable_model_is_skipped(self):
        """Test that unavailable models are skipped during analysis"""
        tracker = get_tracker()
        tracker.reset()
        
        # Mark qwen2vl as unavailable
        tracker.record_failure('qwen2vl')
        tracker.record_failure('qwen2vl')
        tracker.record_failure('qwen2vl')
        
        assert not tracker.is_available('qwen2vl')
        
        # The analyze_with_qwen2vl function should check availability
        # and return None without attempting analysis
        # (This would be tested in actual integration with real models)
    
    def test_available_model_after_cooldown(self):
        """Test that model becomes available after cooldown"""
        tracker = get_tracker()
        tracker.reset()
        tracker.COOLDOWN_SECONDS = 1  # Short cooldown for testing
        
        # Mark unavailable
        tracker.record_failure('nemotron')
        tracker.record_failure('nemotron')
        tracker.record_failure('nemotron')
        
        assert not tracker.is_available('nemotron')
        
        # Wait for cooldown
        import time
        time.sleep(1.1)
        
        # Should be available again
        assert tracker.is_available('nemotron')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
