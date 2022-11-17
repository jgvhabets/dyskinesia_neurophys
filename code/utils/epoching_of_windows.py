"""
Utility-Functions used for to create
epochs (ms level) of exisiting 
windows (sec level) during signal-
processing (e.g. multi-var conn)
"""
# Import public packages
import numpy as np
from dataclasses import dataclass, field
from array import array
from typing import Any


def split_windows_in_epochs():

    return