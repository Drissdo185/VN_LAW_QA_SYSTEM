from typing import Optional, Dict, Any
import logging
from functools import wraps
import time
import asyncio
from dataclasses import dataclass
from log.logging_config import setup_logging

# Setup logging with console output only
setup_logging(
    level=logging.INFO,
    log_format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s'
    # Removed log_file parameter
)

# Rest of utils.py remains the same
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Track performance metrics for operations"""
    start_time: float
    end_time: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Get operation duration in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return 0.0

def measure_performance(func):
    """Decorator to measure function performance"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        metrics = PerformanceMetrics(start_time=time.time())
        try:
            result = await func(*args, **kwargs)
            metrics.end_time = time.time()
            logger.info(
                f"Function {func.__name__} completed in {metrics.duration:.2f} seconds"
            )
            return result
        except Exception as e:
            metrics.end_time = time.time()
            metrics.error = str(e)
            logger.error(
                f"Error in {func.__name__}: {str(e)}. "
                f"Duration: {metrics.duration:.2f} seconds"
            )
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        metrics = PerformanceMetrics(start_time=time.time())
        try:
            result = func(*args, **kwargs)
            metrics.end_time = time.time()
            logger.info(
                f"Function {func.__name__} completed in {metrics.duration:.2f} seconds"
            )
            return result
        except Exception as e:
            metrics.end_time = time.time()
            metrics.error = str(e)
            logger.error(
                f"Error in {func.__name__}: {str(e)}. "
                f"Duration: {metrics.duration:.2f} seconds"
            )
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

class RAGException(Exception):
    """Base exception class for RAG system"""
    pass

class DomainMismatchError(RAGException):
    """Raised when question domain doesn't match current domain"""
    pass

class DocumentRetrievalError(RAGException):
    """Raised when document retrieval fails"""
    pass

class ModelInferenceError(RAGException):
    """Raised when model inference fails"""
    pass

def format_error_response(error: Exception) -> Dict[str, Any]:
    """Format error response with consistent structure"""
    return {
        "error": str(error),
        "error_type": error.__class__.__name__,
        "token_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
    }

def safe_json_serialize(obj: Any) -> Dict:
    """Safely serialize objects to JSON-compatible dictionary"""
    if hasattr(obj, '__dict__'):
        return {
            key: safe_json_serialize(value)
            for key, value in obj.__dict__.items()
            if not key.startswith('_')
        }
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {
            key: safe_json_serialize(value)
            for key, value in obj.items()
        }
    return obj