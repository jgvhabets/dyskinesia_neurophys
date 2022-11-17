

"""
Run Merging data frames

HERE ONLY EPHYS; NO ACC MERGING YET
"""

# import public functions
import sys
from dataclasses import dataclass, field

# import own functions
from lfpecog_features import feats_read_proc_data as read_data_funcs
from utils.utils_fileManagement import get_project_path

# import other functions
import hackathon_mne_mvc as multiVarConn


@dataclass(init=True, repr=True, )
class subjectData:
    """
    Creates Class with all data of one subject,
    stored per datatype. Data is preprocessed and
    ordered by relative time to L-Dopa-intake.

    Input:
        - sub (str): subject code
        - data_version (str): e.g. v2.2
        - project_path (str): main project-directory
            where data is stored 
    """
    sub: str
    data_version: str
    project_path: str
    dType_excl: list = field(default_factory=lambda: [])

    def __post_init__(self,):

        self.dtypes, self.nameFiles, self.dataFiles, sub_path = read_data_funcs.find_proc_data(
            sub=self.sub,
            version=self.data_version,
            project_path=self.project_path
        )

        if len(self.dType_excl) > 0:
            dType_remove = []
            for dType in self.dtypes:
                if dType in self.dType_excl:
                    dType_remove.append(dType)
            [self.dtypes.remove(d) for d in dType_remove]
        
        for dType in self.dtypes:

            setattr(
                self,
                dType,
                read_data_funcs.dopaTimedDf(
                    dType,
                    self.nameFiles,
                    self.dataFiles,
                    sub_path,
                )
            )



if __name__ == '__main__':

    data = subjectData(
        sub=sys.argv[1],
        data_version='v3.0',
        project_path=get_project_path()
    )
    print(f'Sub-merged DATA SHAPE: {data.lfp_right.data.shape}')
    print(f'column names are {data.lfp_right.data.keys()}')

    print(vars(multiVarConn).keys())