"""
Function to store MNE obects as pickle
"""

# import puiblic packages
from os.path import join
import numpy as np
from dataclasses import dataclass, field
from typing import Any

# import own functions
from utils.utils_fileManagement import get_project_path



@dataclass(init=True, repr=True)
class pickle_EpochedArrays:

    mne_object: Any = None
    list_mne_objects: list = field(default_factory=list)
    window_times: list = field(default_factory=list)
    
    def __post_init__(self,):

        if len(self.list_mne_objects) >= 1:
            self.info = self.list_mne_objects[0].info
        
        else:
            self.info = self.mne_object.info
