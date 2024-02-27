"""
Created By  : ...
Created Date: DD/MM/YYYY
Description : ...
"""
import logging


def get_logger(app_name):
    """Create and configure logger.

    :return: logger
    :rtype: Logger
    """
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
