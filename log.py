"""
A workaround to create a single logfile with datetime info.
The logfile is shared among all scripts.
Delete TMP_DATETIME_FILE before running pipeline.
TODO: Use Singleton pattern implementation
"""

import datetime
import logging
import os
import time

from basic_config import LOG_DIR, TMP_DATETIME_FILE


TIMESTAMP = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S_%f")

with open(TMP_DATETIME_FILE, "a+") as f:
    f.write(TIMESTAMP + "\n")


def get_first_timestamp():
    """
    Utility function to get the time stamp
    when experiment starts
    """
    with open(TMP_DATETIME_FILE, "r") as f:
        timestamp = f.readline()

    return timestamp.replace("\n", "")


os.makedirs(LOG_DIR, exist_ok=True)

# Config logger
logging.basicConfig(
    filename=f"{LOG_DIR}/exp_{get_first_timestamp()}.log",
    filemode="a",
    # format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    format="%(message)s",
    datefmt="%Y-%m-%d_%H-%M-%S",
    level=logging.INFO,
)

logger = logging.getLogger()
