'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                        Get a Variable of Interest for a particular session                            -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import matplotlib.pyplot as plt; import numpy as np
from analysis_funcs import get_particular_variables, filter_features



''' OPTIONS FOR EXTRACTION:
'speed', 'velocity relative to shelter', 'head direction relative to shelter', 'head turn relative to shelter', 'distance from shelter', 'distance from center', 'in shelter', 'position', 'behavioural cluster'
'''


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
save_folder_location = "C:\\Drive\\Video Analysis\\data\\baseline_analysis\\test_pipeline\\"
session_name_tag = 'loom_1'







# ---------------------------
# Select analysis parameters
# ---------------------------
variables_of_interest = ['speed', 'velocity relative to shelter', 'head direction relative to shelter', 
                         'head turn relative to shelter', 'distance from shelter', 'distance from center', 'in shelter', 'position', 'behavioural cluster']


#variables_of_interest = ['behavioural cluster']

clip_suprious_values = True
filter_variables = False
filter_length = 3
sigma = 2




# variables relating to the behavioural clustering
data_library_name_tag = 'streamlined'
model_name_tag = '4PC'
model_sequence = False
cluster_names = ['groom, hunch, rear', 'outstretch' , 'turn / investigate' , 'locomote', 'pause from locomotion'] # just for reference




# -----
# do it
# -----

variables_of_interest_matrix = get_particular_variables(variables_of_interest, save_folder_location, data_library_name_tag, session_name_tag, model_name_tag, model_sequence, clip_suprious_values)
print('done')

if filter_variables:
    variables_of_interest_matrix_filtered = filter_features(variables_of_interest_matrix , filter_length, sigma)
    variables_of_interest_matrix_filtered[np.isnan(variables_of_interest_matrix_filtered)] = variables_of_interest_matrix[np.isnan(variables_of_interest_matrix_filtered)]


                
            









