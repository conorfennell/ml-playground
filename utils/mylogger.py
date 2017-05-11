"""
logging support
"""
from __future__ import absolute_import
import logging
import logging.config
import os

logging.config.fileConfig(os.getcwd()+ '/logging.conf')

def get_logger(name):
    """Get instance of logger"""
    return logging.getLogger(name)
