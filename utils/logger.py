import logging
import sys

RED = "\033[91m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"
BOLD = "\033[1m"

def configure_logging(dev_mode: bool = False, module_name: str = "transforms_class"):
    """
    Configure logging for the module and its submodules.
    
    Parameters
    ----------
    dev_mode : bool
        If True, logging level = DEBUG, else INFO
    module_name : str
        Name of the module to configure (default "transforms_class")
    """
    logger = logging.getLogger(module_name)
    
    if dev_mode:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # Optional: add a handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)  # stdout instead of stderr (because of logging package behavior)
        formatter = logging.Formatter(
            '%(asctime)s %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger