'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                  Perform PCA on wavelet-transformed 3D mouse video                             -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np;
from matplotlib import pyplot as plt;
from scipy import linalg;
import os;
import sklearn;
from sklearn.externals import joblib
from sklearn import mixture;
from sklearn.model_selection import KFold;
from hmmlearn import hmm;
import warnings;
from learning_funcs import cross_validate_model, create_sequential_data, prepare_data_for_model
from learning_funcs import filter_features, calculate_and_save_model_output, set_up_PC_cluster_plot, create_legend

warnings.filterwarnings('ignore')

''' -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------              Select data file and analysis parameters                 --------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------'''

# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
save_folder_location = "C:\\Drive\\Video Analysis\\data\\baseline_analysis\\test_pipeline\\"
session_name_tags = ['loom_1', 'loom_2', 'loom_3', 'loom_4', 'loom_5', 'loom_6',
                     'clicks_1', 'clicks_2', 'clicks_3',
                     'post_clicks_1', 'post_clicks_2', 'post_clicks_3']
data_library_name_tag = 'streamlined'

model_name_tag = '10PC'
model_sequence = False

do_not_overwrite = False
generate_new_model = True

# ---------------------------
# Select analysis parameters
# ---------------------------
# Clustering Settings
model_type = 'hmm'  # hmm or gmm
num_clusters = 5; num_PCs_used = 10
use_speed_as_pseudo_PC = True
use_angular_speed_as_pseudo_PC = True

# scale down velocity's importance relative to PC1
vel_scaling_factor = 1; turn_scaling_factor = 1



if model_sequence:
    window_size, windows_to_look_at = 6, 1
else:
    window_size, windows_to_look_at = 0, 0

# Smoothing Settings
filter_data_for_model = True
filter_length = 3  # half-width of gaussian smoothing
sigma = 2  # standard deviation in frames of gaussian filter




# --------------------------------
# Select more analysis parameters
# --------------------------------
# Cross-Validation Settings
cross_validation = False
# K-fold cross-val // random seeds // how many clusters // error upper bound
K_range = [10]; seed_range = [0]; num_clusters_range = [3, 5]; tol = .001

# Misc Settings
plot_result = True; num_PCs_shown = 3; seed = 3; show_unchosen_cluster = .15
























''' -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------              Prepare data and do cross-validation                       --------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------'''

# -------------------
# Load relevant data
# -------------------
folder_location_data_library = save_folder_location + data_library_name_tag + '\\'
file_location_data_library = folder_location_data_library + data_library_name_tag


if model_sequence:  # modify saved file names depending on type of clustering:
    model_type_and_name_tag = 'seq_' + model_name_tag
else:
    model_type_and_name_tag = model_name_tag


if generate_new_model:
    print('preparing data...')
    if not os.path.isdir(folder_location_data_library):
        os.makedirs(folder_location_data_library)
        print("saving to " + folder_location_data_library)
    np.save(file_location_data_library + '_session_name_tags.npy',session_name_tags)
    
    
    # load position-orientation-velocity
    # create huge position_orientation_velocity  and pca coefficient arrays for all sessions
    pca = joblib.load(file_location_data_library + '_pca')
    pca_coeffs = np.zeros((1,pca.n_components_))
    position_orientation_velocity = np.array(([], [], [], [], [], [], [], [])).T
    disruptions = np.array([]); out_of_bounds = []
    
    for session in enumerate(session_name_tags):
        if os.path.isfile(save_folder_location + session[1] + '\\' + session[1] + '_position_orientation_velocity_corrected.npy'):
            position_orientation_velocity_cur = np.load(save_folder_location + session[1] + '\\' + session[1] + '_position_orientation_velocity_corrected.npy')
        else:
            position_orientation_velocity_cur = np.load(save_folder_location + session[1] + '\\' + session[1] + '_position_orientation_velocity.npy')
            print('loaded non-flip-corrected data')
    
        # keep track of which indices are in-bounds and valid
        out_of_bounds_cur = position_orientation_velocity_cur[:,1]
        disruptions_cur = np.ones(len(out_of_bounds_cur)).astype(bool)
        disruptions_cur[1:] = np.not_equal(out_of_bounds_cur[1:],out_of_bounds_cur[:-1])
        disruptions_cur = disruptions_cur[out_of_bounds_cur==0]
        
        position_orientation_velocity_cur = position_orientation_velocity_cur[out_of_bounds_cur==0,:]
        position_orientation_velocity = np.concatenate((position_orientation_velocity, \
                                                        position_orientation_velocity_cur), axis=0)
        disruptions = np.concatenate((disruptions, disruptions_cur)).astype(bool)
        out_of_bounds.append(out_of_bounds_cur)
        
        # load PCs
        pca_coeffs_cur = np.load(save_folder_location + session[1] + '\\' + session[1] + '_pca_coeffs_' + data_library_name_tag + '.npy')
        pca_coeffs = np.concatenate((pca_coeffs, pca_coeffs_cur), axis=0)
    
    pca_coeffs = pca_coeffs[1:,:]     
    
    data = prepare_data_for_model(pca_coeffs, num_PCs_used, position_orientation_velocity, disruptions, \
                               vel_scaling_factor, turn_scaling_factor, use_speed_as_pseudo_PC, use_angular_speed_as_pseudo_PC, \
                               filter_data_for_model, filter_length, sigma, model_sequence, window_size, windows_to_look_at)

    
    # ------------
    # Save settings
    # -------------  
    np.save(file_location_data_library + '_' + model_type + '_settings_' + model_type_and_name_tag + '.npy',  # save settings
            [use_speed_as_pseudo_PC, True, False, use_angular_speed_as_pseudo_PC, num_PCs_used, window_size, windows_to_look_at, np.max(data)])
    
    # --------------------
    # Do cross-validation
    # --------------------
    if cross_validation:
        # xval_scores gives: % in cluster w/ high confidence; 10th percentile of confidence; score; bayesian info content (currently only for GMM)
        xval_scores = cross_validate_model(data, model_type, K_range, seed_range, num_clusters_range, tol)
    
    
    
    
    
    
    
    
    ''' -------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------              Generate Model                            --------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------'''
    
    # -------------------------
    # Initialize mixture model
    # -------------------------
    if model_type == 'hmm':
        model = hmm.GaussianHMM(n_components=num_clusters, covariance_type="full", algorithm='viterbi', tol=.00001,
                                random_state=seed)
    elif model_type == 'gmm':
        model = mixture.GaussianMixture(n_components=num_clusters, tol=.00001, covariance_type='full', random_state=seed)
    
    # ---------------------------
    # Fit and save mixture model
    # ---------------------------
    print('fitting model...')
    model.fit(data)  # fit model
    if os.path.isfile(file_location_data_library + '_' + model_type + '_' + model_type_and_name_tag) and do_not_overwrite:
        raise Exception('File already exists')
    joblib.dump(model, file_location_data_library + '_' + model_type + '_' + model_type_and_name_tag)
else:
    model = joblib.load(file_location_data_library + '_' + model_type + '_' + model_type_and_name_tag)
    
    use_speed_as_pseudo_PC, _, _, use_angular_speed_as_pseudo_PC, num_PCs_used, window_size, windows_to_look_at, _ = \
            np.load(file_location_data_library + '_' + model_type + '_settings_' + model_type_and_name_tag + '.npy').astype(int) # save settings



# -------------------------------------------------
# Calculate and save model output for each session
# -------------------------------------------------
print('saving model output...')
for session in enumerate(session_name_tags):
    file_location_data_cur = save_folder_location + session[1] + '\\' + session[1]

    if os.path.isfile(file_location_data_cur  + '_position_orientation_velocity_corrected.npy'):
        position_orientation_velocity_cur = np.load(file_location_data_cur + '_position_orientation_velocity_corrected.npy')
    else:
        position_orientation_velocity_cur = np.load(file_location_data_cur + '_position_orientation_velocity.npy')

    # keep track of which indices are in-bounds and valid
    out_of_bounds_cur = position_orientation_velocity_cur[:, 1]
    disruptions_cur = np.ones(len(out_of_bounds_cur)).astype(bool)
    disruptions_cur[1:] = np.not_equal(out_of_bounds_cur[1:], out_of_bounds_cur[:-1])
    disruptions_cur = disruptions_cur[out_of_bounds_cur == 0]
    position_orientation_velocity_cur = position_orientation_velocity_cur[out_of_bounds_cur == 0, :]

    # load PCs
    pca_coeffs_cur = np.load(file_location_data_cur + '_pca_coeffs_' + data_library_name_tag + '.npy')

    data = prepare_data_for_model(pca_coeffs_cur, num_PCs_used, position_orientation_velocity_cur, disruptions_cur, \
                                  vel_scaling_factor, turn_scaling_factor, use_speed_as_pseudo_PC,
                                  use_angular_speed_as_pseudo_PC, filter_data_for_model, 
                                  filter_length, sigma, model_sequence, window_size, windows_to_look_at)

    calculate_and_save_model_output(data, model, num_clusters, file_location_data_cur, model_type, \
                                    model_type_and_name_tag, do_not_overwrite)
    
