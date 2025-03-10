import logging
import sys
from typing import Optional

def setup_logging(
    level: int = logging.INFO,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Set up logging configuration for the entire application.
    Only logs to console regardless of log_file parameter.
    """
    if log_format is None:
        log_format = '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
    
    
    formatter = logging.Formatter(log_format)
    
   
    handlers = []
    
  
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    
    for handler in root_logger.handlers[:]:
        
        try:
            handler.close()
        except:
            pass
        root_logger.removeHandler(handler)
    
    
    for handler in handlers:
        root_logger.addHandler(handler)