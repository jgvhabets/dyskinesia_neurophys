"""
Python main script which should be ran from
command-line to create data arrays ordered
by dopa-time of full subject recordings
"""

if __name__ == '__main__':
    # only run script if called from cmnd-line

    # Import packages and functions
    from os.path import join, dirname
    from os import getcwd
    import sys
    import json

    proj_path = getcwd()  # should be /../dyskinesia_neurophys
    if proj_path[-4:] == 'code':
        proj_path = dirname(proj_path)

    ft_path = join(proj_path, 'code/lfpecog_features')
    sys.path.append(ft_path)  # enable import in sub-functions

    # Import own functions
    import feats_read_data
   
    # open argument (json file) defined in command (line)
    with open(sys.argv[1], 'r') as json_data:
    
        settings = json.load(json_data)  # gets dir

        print(settings)

    for sub in settings['subs_include']:

        dtypes, nameFiles, dataFiles, sub_path = feats_read_data.find_proc_data(
            sub=sub,
            version=settings['data_version'],
            project_path=settings['project_path']
        )

        print(sub, dtypes)

        for dType in dtypes:

            if dType not in settings['data_include']: continue

            dopa_array = feats_read_data.create_dopa_timed_array(
                dType, nameFiles, dataFiles, sub_path
            )            
            print(dopa_array)
