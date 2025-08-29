"""
Production-grade retry and error recovery utilities
AutoResolve v3.2 Enterprise Reliability Module
"""

import time
import random
import logging
from functools import wraps
from typing import Callable, Any, Type, Tuple

logger = logging.getLogger(__name__)

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass

class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def _should_attempt_reset(self) -> bool:
        return (self.state == 'OPEN' and 
                self.last_failure_time and
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        # Check if we should attempt reset
        if self._should_attempt_reset():
            self.state = 'HALF_OPEN'
            logger.info(f"Circuit breaker {func.__name__}: HALF_OPEN (attempting reset)")
        
        # Block if circuit is open
        if self.state == 'OPEN':
            raise CircuitBreakerError(f"Circuit breaker open for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == 'HALF_OPEN':
                logger.info(f"Circuit breaker {func.__name__}: CLOSED (reset successful)")
                self.state = 'CLOSED'
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Check if we should open the circuit
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                logger.error(f"Circuit breaker {func.__name__}: OPEN (failures: {self.failure_count})")
            
            raise e

def exponential_backoff_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    jitter: bool = True
):
    """Decorator for exponential backoff retry with jitter"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries: {e}")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                
                except Exception as e:
                    # Don't retry on unexpected exceptions
                    logger.error(f"{func.__name__} failed with unexpected error: {e}")
                    raise e
            
            # Should never reach here, but just in case
            raise last_exception
        
        return wrapper
    return decorator

def timeout_retry(timeout: float = 30.0, retries: int = 2):
    """Decorator for operations that might hang"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"{func.__name__} exceeded {timeout}s timeout")
            
            for attempt in range(retries + 1):
                try:
                    # Set timeout alarm
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(timeout))
                    
                    try:
                        result = func(*args, **kwargs)
                        signal.alarm(0)  # Cancel alarm
                        return result
                    finally:
                        signal.alarm(0)  # Ensure alarm is cancelled
                        
                except TimeoutError as e:
                    if attempt == retries:
                        logger.error(f"{func.__name__} timed out after {retries} attempts")
                        raise e
                    logger.warning(f"{func.__name__} timed out on attempt {attempt + 1}, retrying...")
                    
            raise TimeoutError(f"{func.__name__} exceeded timeout after {retries} retries")
        
        return wrapper
    return decorator

def graceful_degradation(fallback_func: Callable = None, fallback_value: Any = None):
    """Decorator for graceful degradation on failures"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"{func.__name__} failed, using fallback: {e}")
                
                if fallback_func:
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback {fallback_func.__name__} also failed: {fallback_error}")
                        return fallback_value
                
                return fallback_value
        
        return wrapper
    return decorator

# Pre-configured retry decorators for common scenarios
network_retry = exponential_backoff_retry(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exceptions=(ConnectionError, requests.RequestException, TimeoutError),
    jitter=True
)

file_operation_retry = exponential_backoff_retry(
    max_retries=2,
    base_delay=0.5,
    max_delay=5.0,
    exceptions=(IOError, OSError),
    jitter=False
)

model_loading_retry = exponential_backoff_retry(
    max_retries=2,
    base_delay=2.0,
    max_delay=10.0,
    exceptions=(RuntimeError, ImportError),
    jitter=False
)

# Global circuit breakers for external services
embedder_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=300)
api_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=120)
filesystem_circuit_breaker = CircuitBreaker(failure_threshold=10, recovery_timeout=60)