# -*- coding: utf-8 -*-
"""
Created the 19/11/2023

@author: Sebastien Weber
"""

from pathlib import Path
from pymodaq_utils.config import BaseConfig


class AlgoConfig(BaseConfig):
    """ Config class to be used as a cache for latest configuration values of the algo"""
    config_template_path = None
    config_name = f"algo2D_settings"

