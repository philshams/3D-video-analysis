'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                        Get a Variable of Interest for a particular session                            -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import matplotlib.pyplot as plt; import numpy as np
from analysis_funcs import get_particular_variables, filter_features



''' OPTIONS FOR EXTRACTION:
'speed', 'angular speed', 'together', 'position', 'behavioural cluster'
'''


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
save_folder_location = "C:\\Drive\\Video Analysis\\data\\3D_pipeline\\"

session_name_tag = 'session1'
twoD = True






# ---------------------------
# Select analysis parameters
# ---------------------------
variables_of_interest = ['speed', 'angular speed', 'together', 'position', 'behavioural cluster']
#variables_of_interest = ['behavioural cluster']

clip_suprious_values = True
filter_variables = False
filter_length = 3
sigma = 2




# variables relating to the behavioural clustering
data_library_name_tag = 'test'
model_name_tag = '6PC'
if twoD:
    model_name_tag = model_name_tag + '2D'
model_sequence = False
cluster_names = ['orthogonal moving' ,'orthogonal still' , 'in a line' , 'in a ball'] # just for reference




# -----
# do it
# -----

variables_of_interest_matrix = get_particular_variables(variables_of_interest, save_folder_location, data_library_name_tag, session_name_tag, model_name_tag, model_sequence, clip_suprious_values)
print('done')

if filter_variables:
    variables_of_interest_matrix_filtered = filter_features(variables_of_interest_matrix , filter_length, sigma)
    variables_of_interest_matrix_filtered[np.isnan(variables_of_interest_matrix_filtered)] = variables_of_interest_matrix[np.isnan(variables_of_interest_matrix_filtered)]


                
            









