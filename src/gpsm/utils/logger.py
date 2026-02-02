from loguru import logger
import os
from datetime import datetime

# Create log directory if it doesn't exist
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

# Generate timestamped log file name
timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
log_file_path = os.path.join(log_dir, f"{timestamp}.log")

# Remove default logger and add our own
logger.remove()  # Remove the default stderr logger
logger.add(
    log_file_path,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
    level="INFO",
    rotation="10 MB",  # Optional: rotate after 10 MB
    retention="10 days",  # Optional: keep logs for 10 days
    compression="zip"  # Optional: compress old logs
)

# Optional: also log to the console
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
)