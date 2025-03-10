import logging
import sys
from typing import Optional

def setup_logging(level: int = logging.INFO, log_format: Optional[str] = None):
    if log_format is None:
        log_format = '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
    
    formatter = logging.Formatter(log_format)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers[:]:
        try:
            handler.close()
        except:
            pass
        root_logger.removeHandler(handler)
    
    root_logger.addHandler(console_handler)