'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                  Perform PCA on wavelet-transformed 3D mouse video                             -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np; from matplotlib import pyplot as plt; from scipy import linalg
from sklearn import mixture; from sklearn.model_selection import KFold; from hmmlearn import hmm; import warnings; warnings.filterwarnings('ignore')
from learning_funcs import cross_validate_model, add_velocity_as_feature, add_pose_change_as_feature, create_sequential_data
from learning_funcs import filter_features, calculate_and_save_model_output, set_up_PC_cluster_plot, create_legend


#%% -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------              Select data file and analysis parameters                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
file_loc = 'C:\Drive\Video Analysis\data\\'
date = '05.02.2018\\'
mouse_session = '202-1a\\'
save_vid_name = 'analyze_2D'

date = '28.02.2018\\'
mouse_session = '205_2a\\'
save_vid_name = 'analyze' # name-tag to be associated with all saved files

file_loc = 'C:\Drive\Video Analysis\data\\'
date = '15.03.2018\\'
mouse_session = 'bj141p2\\'
save_vid_name = 'rectified_norm' # name-tag to be associated with all saved files

file_loc = file_loc + date + mouse_session + save_vid_name

# ---------------------------
# Select analysis parameters
# ---------------------------
# Clustering Settings
model_type = 'hmm' #hmm or gmm
num_clusters = 4 #number of poses
num_PCs_used = 4 #number of PCs used as features
add_velocity = True #include velocity as a pseudo-PC
vel_scaling_factor = 2 #scale down velocity's importance relative to PC1
add_change = False #include that change in PC-space since last frame as a pseudo-PC (recommended only for trajectory)
speed_only = True
video_type = 'justone'

# Trajectory Settings
model_sequence = False
if model_sequence:
    window_size = 3 #frames
    windows_to_look_at = 3
else:
    window_size, windows_to_look_at = 0,0

# Smoothing Settings
filter_data_for_model = True
filter_length = 5 # total width of gaussian smoothing
sigma = 2 #standard deviation in frames of gaussian filter


# --------------------------------
# Select more analysis parameters
# --------------------------------
# Cross-Validation Settings
cross_validation = False
K_range = [10] # K-fold cross-validation
seed_range = [0] # random seed used
num_clusters_range = [3,5] # range of how many clusters used in hmm
tol = .001 # how low of an error needed to end model improvement

# Misc Settings
plot_result = True
num_PCs_shown = 3
seed = 1
show_unchosen_cluster = .15 #if unchosen cluster has this proportion responsibility, plot it too
do_not_overwrite = False



#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Prepare data and do cross-validation                       --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# -------------------
# Load relevant data
# -------------------
velocity = np.load(file_loc + '_velocity.npy')
try:
    disruptions = np.load(file_loc + '_disruption.npy')
except:
    disruptions = []
if video_type == '2D':
    file_loc = file_loc +'_2D'
pca_coeffs = np.load(file_loc + '_pca_coeffs.npy')


data_for_model = pca_coeffs[:,0:num_PCs_used]
#modify saved file names depending on type of clustering:
if model_sequence: 
    suffix = '_seq'
else:
    suffix = ''


# --------------------------------
# Include velocity as a pseudo-PC
# --------------------------------
if add_velocity:
    if add_change: #if using add_change, then apply speed_only
        speed_only = True
    data_for_model = add_velocity_as_feature(data_for_model, speed_only, velocity, vel_scaling_factor,disruptions)

# -------------------------------------
# Smooth features going into the model
# -------------------------------------
if filter_data_for_model:
    data_for_model = filter_features(data_for_model, filter_length, sigma) #apply gaussian smoothing


# --------------------------------------
# Include change in pose as a pseudo-PC
# --------------------------------------
if add_change:
    data_for_model = add_pose_change_as_feature(data_for_model, vel_scaling_factor, num_PCs_used)  
    
data = data_for_model #use this for the model, unless modeling sequence


# ------------------------------------------------
# Create array of moving-window-chunked sequences
# ------------------------------------------------
if model_sequence:  #add feature chunks preceding and following the frame in question
    data_for_model_sequence = create_sequential_data(data_for_model, window_size, windows_to_look_at)

    data = data_for_model_sequence#use this for the model, if modeling sequence


# -------------
# Save settings
# -------------  
np.save(file_loc + '_' + model_type + '_settings' + suffix + '.npy',  #save settings
        [add_velocity, speed_only, add_change, num_PCs_used, window_size, windows_to_look_at, np.max(data_for_model)])



# --------------------
# Do cross-validation
# --------------------
if cross_validation:
    xval_scores = cross_validate_model(data, model_type, K_range, seed_range, num_clusters_range, tol)
    #xval_scores gives: % in cluster w/ high confidence; 10th percentile of confidence; score; bayesian info content (currently only for GMM)
 
    


#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Generate Gaussian Mixture Model                            --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# -------------------------
# Initialize mixture model
# -------------------------
if model_type == 'hmm':
    model = hmm.GaussianHMM(n_components=num_clusters, covariance_type="full",algorithm='viterbi',tol=.00001,random_state=seed)
elif model_type == 'gmm':
    model = mixture.GaussianMixture(n_components=num_clusters,tol=.00001,covariance_type='full',random_state=seed)
    
    
# ---------------------------
# Fit and save mixture model
# ---------------------------
model.fit(data)   #fit model


# -------------------------------
# Calculate and save model output
# -------------------------------
calculate_and_save_model_output(data, model, num_clusters, file_loc, model_type, suffix, do_not_overwrite)



#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Plot PCs and Clusters over Time                            --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

if plot_result:
    plt.close('all')
    #reload relevant data
    components_binary = np.load(file_loc + '_components_binary' + suffix + '.npy')
    unchosen_components_binary = np.load(file_loc + '_unchosen_components_binary' + suffix + '.npy')
    unchosen_probabilities = np.load(file_loc + '_unchosen_probabilities' + suffix + '.npy')
    
    # set up plot
    plot_colors = ['red','deepskyblue','green','blueviolet','saddlebrown','lightpink','yellow','white']
    set_up_PC_cluster_plot(figsize=(30,10), xlim=[0,2000])
    
    # plot PCs
    data_for_model_normalized = data_for_model / np.max(data_for_model) #set max to 1 for visualization purposes
    np.save(file_loc+'_data_for_' + model_type + '_normalized', data_for_model_normalized)
    plt.plot(data_for_model_normalized[:,0:num_PCs_shown])
    
    # plot velocity and/or pose change
    if add_velocity:
        plt.plot(data_for_model_normalized[:,-2-add_change+speed_only], color = 'k',linewidth=2) #plot speed
        if not(speed_only) or add_change: #plot turn speed or, if available, pose change
            plt.plot(data_for_model_normalized[:,-1], color = 'gray', linestyle = '--',linewidth=2)
      
    # plot raster of clusters above PCs
    for n in range(num_clusters):
        #plot chosen pose:
        component_frames = find(components_binary[:,n])
        plt.scatter(component_frames,np.ones(len(component_frames))*1,color=plot_colors[n],alpha=.7,marker='|',s=700)
        #plot 2nd-place pose:
        unchosen_component_frames = find(unchosen_components_binary[:,n] * (unchosen_probabilities>show_unchosen_cluster))
        plt.scatter(unchosen_component_frames,np.ones(len(unchosen_component_frames))*.95,color=plot_colors[n],alpha=.4,marker='|',s=700)
    
    # Create legend
    legend_entries = create_legend(num_PCs_shown, add_velocity, speed_only, add_change)
    

