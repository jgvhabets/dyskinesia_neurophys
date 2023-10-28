"""
Defining Class with Dummy data
to load via pickle
"""

# import packages
from dataclasses import dataclass
from numpy import ndarray


@dataclass(repr=True, init=True,)
class DummyData_Base:
    """
    Create Dummy Data
    e.g. for timeflux testing
    """
    stn: ndarray
    stn_name: str
    ecog: ndarray
    ecog_name: str
    fs: int