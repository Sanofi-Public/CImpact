"""
Created By  : ...
Created Date: DD/MM/YYYY
Description : ...
"""

import argparse
from utils.logging import get_logger


APP_NAME = 'MyProject'
LOGGER = get_logger(APP_NAME)


def dummy(dum):
    """Example function

    :param dum: Text to log.
    :type number: str
    :return: The entry text.
    :rtype: str
    """
    LOGGER.info(f'{dum} in progress')
    return dum


def main():
    """Main process
    """
    dum = dummy(args.process)
    LOGGER.info(f'{dum} completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', type=str, default='test')
    args = parser.parse_args()
    main()
