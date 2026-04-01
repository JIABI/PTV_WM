import logging

def setup_logging(level: str = "INFO") -> None:
    """Setup global logging."""
    logging.basicConfig(level=getattr(logging, level.upper()), format="[%(levelname)s] %(message)s")
