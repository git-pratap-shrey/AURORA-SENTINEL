"""
Model Availability Tracker
Implements circuit breaker pattern for AI models
"""
import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ModelStatus:
    """Status information for a model"""
    name: str
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    is_available: bool = True
    total_failures: int = 0
    total_successes: int = 0
    last_success_time: Optional[float] = None


class ModelAvailabilityTracker:
    """
    Tracks model availability and implements circuit breaker pattern.
    
    - Marks model unavailable after 3 consecutive failures
    - Skips unavailable models for 5 minutes before retrying
    - Provides status reporting for health checks
    """
    
    # Configuration
    MAX_CONSECUTIVE_FAILURES = 3
    COOLDOWN_SECONDS = 300  # 5 minutes
    
    def __init__(self):
        """Initialize tracker with empty status for all models"""
        self._models: Dict[str, ModelStatus] = {
            'qwen2vl': ModelStatus(name='Qwen2-VL'),
            'nemotron': ModelStatus(name='Nemotron'),
            'ollama': ModelStatus(name='Ollama')
        }
    
    def record_success(self, model_name: str) -> None:
        """
        Record a successful model execution.
        Resets consecutive failures and marks model as available.
        
        Args:
            model_name: Name of the model ('qwen2vl', 'nemotron', 'ollama')
        """
        if model_name not in self._models:
            logger.warning(f"Unknown model: {model_name}")
            return
        
        status = self._models[model_name]
        status.consecutive_failures = 0
        status.is_available = True
        status.total_successes += 1
        status.last_success_time = time.time()
        
        logger.debug(f"[{status.name}] Success recorded (total: {status.total_successes})")
    
    def record_failure(self, model_name: str, error: Optional[Exception] = None) -> None:
        """
        Record a model failure.
        Increments consecutive failures and marks unavailable after threshold.
        
        Args:
            model_name: Name of the model ('qwen2vl', 'nemotron', 'ollama')
            error: Optional exception that caused the failure
        """
        if model_name not in self._models:
            logger.warning(f"Unknown model: {model_name}")
            return
        
        status = self._models[model_name]
        status.consecutive_failures += 1
        status.total_failures += 1
        status.last_failure_time = time.time()
        
        # Mark unavailable after threshold
        if status.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            status.is_available = False
            logger.warning(
                f"[{status.name}] Marked UNAVAILABLE after {status.consecutive_failures} "
                f"consecutive failures. Will retry in {self.COOLDOWN_SECONDS}s"
            )
        else:
            logger.warning(
                f"[{status.name}] Failure {status.consecutive_failures}/{self.MAX_CONSECUTIVE_FAILURES}"
                + (f": {error}" if error else "")
            )
    
    def is_available(self, model_name: str) -> bool:
        """
        Check if a model is available for use.
        
        If model was marked unavailable, checks if cooldown period has elapsed.
        If cooldown elapsed, marks model as available for retry.
        
        Args:
            model_name: Name of the model ('qwen2vl', 'nemotron', 'ollama')
            
        Returns:
            True if model is available, False if in cooldown period
        """
        if model_name not in self._models:
            logger.warning(f"Unknown model: {model_name}")
            return False
        
        status = self._models[model_name]
        
        # If already available, return True
        if status.is_available:
            return True
        
        # Check if cooldown period has elapsed
        if status.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - status.last_failure_time
        
        if time_since_failure >= self.COOLDOWN_SECONDS:
            # Cooldown elapsed, allow retry
            logger.info(
                f"[{status.name}] Cooldown period elapsed ({time_since_failure:.0f}s). "
                "Allowing retry..."
            )
            status.is_available = True
            status.consecutive_failures = 0  # Reset for fresh start
            return True
        
        # Still in cooldown
        remaining = self.COOLDOWN_SECONDS - time_since_failure
        logger.debug(
            f"[{status.name}] Still unavailable. Retry in {remaining:.0f}s"
        )
        return False
    
    def get_status(self, model_name: str) -> Optional[Dict]:
        """
        Get detailed status for a specific model.
        
        Args:
            model_name: Name of the model ('qwen2vl', 'nemotron', 'ollama')
            
        Returns:
            Dictionary with status information or None if model unknown
        """
        if model_name not in self._models:
            return None
        
        status = self._models[model_name]
        
        # Calculate time since last failure
        time_since_failure = None
        cooldown_remaining = None
        if status.last_failure_time:
            time_since_failure = time.time() - status.last_failure_time
            if not status.is_available:
                cooldown_remaining = max(0, self.COOLDOWN_SECONDS - time_since_failure)
        
        # Calculate time since last success
        time_since_success = None
        if status.last_success_time:
            time_since_success = time.time() - status.last_success_time
        
        return {
            'name': status.name,
            'available': status.is_available,
            'consecutive_failures': status.consecutive_failures,
            'total_failures': status.total_failures,
            'total_successes': status.total_successes,
            'last_failure_time': datetime.fromtimestamp(status.last_failure_time).isoformat() if status.last_failure_time else None,
            'last_success_time': datetime.fromtimestamp(status.last_success_time).isoformat() if status.last_success_time else None,
            'time_since_failure_seconds': round(time_since_failure, 1) if time_since_failure else None,
            'time_since_success_seconds': round(time_since_success, 1) if time_since_success else None,
            'cooldown_remaining_seconds': round(cooldown_remaining, 1) if cooldown_remaining else None
        }
    
    def get_all_status(self) -> Dict[str, Dict]:
        """
        Get status for all tracked models.
        
        Returns:
            Dictionary mapping model names to their status information
        """
        return {
            model_name: self.get_status(model_name)
            for model_name in self._models.keys()
        }
    
    def reset(self, model_name: Optional[str] = None) -> None:
        """
        Reset tracking for a model or all models.
        
        Args:
            model_name: Name of model to reset, or None to reset all
        """
        if model_name:
            if model_name in self._models:
                self._models[model_name] = ModelStatus(name=self._models[model_name].name)
                logger.info(f"[{model_name}] Status reset")
        else:
            for name in self._models.keys():
                self._models[name] = ModelStatus(name=self._models[name].name)
            logger.info("All model statuses reset")


# Global singleton instance
_tracker = ModelAvailabilityTracker()


def get_tracker() -> ModelAvailabilityTracker:
    """Get the global model availability tracker instance"""
    return _tracker
