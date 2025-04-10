from common.config_static import LOGS_DIR
from datetime import datetime
import sys
import logging
import os

logger = logging.getLogger("MLOpsPipeline")


def init_logger():
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f"{LOGS_DIR}/mlops_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger.info("Logger initialized")
