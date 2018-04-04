'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                  Perform PCA on wavelet-transformed 3D mouse video                             -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np; from matplotlib import pyplot as plt; from scipy import linalg; import os; import sklearn; from sklearn.externals import joblib
from sklearn import mixture; from sklearn.model_selection import KFold; from hmmlearn import hmm; import warnings; warnings.filterwarnings('ignore')
from learning_funcs import cross_validate_model, add_velocity_as_feature, add_pose_change_as_feature, create_sequential_data
from learning_funcs import filter_features, calculate_and_save_model_output, set_up_PC_cluster_plot, create_legend


#%% -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------              Select data file and analysis parameters                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
file_location = 'C:\Drive\Video Analysis\data\\'
data_folder = 'baseline_analysis\\'
analysis_folder = 'together_for_model\\'

concatenated_data_name_tag = 'analyze3' 
model_name_tag = '15_4'
# 11 is 3PCs, 14, 15, 16 are 4,5,6 PCs; 15nt, 18nt for no turn

# ---------------------------
# Select analysis parameters
# ---------------------------
# Clustering Settings
model_type = 'hmm' #hmm or gmm
num_clusters = 4 #number of poses
num_PCs_used = 4 #number of PCs used as features
add_velocity = True #include velocity as a pseudo-PC
add_turn = True
vel_scaling_factor = .5 #scale down velocity's importance relative to PC1
turn_scaling_factor = .75 #scale down velocity's importance relative to PC1


# Trajectory Settings
model_sequence = True
if model_sequence:
    window_size = 10 #frames
    windows_to_look_at = 1
else:
    window_size, windows_to_look_at = 0,0

# Smoothing Settings
filter_data_for_model = True
filter_length = 3 # half-width of gaussian smoothing
sigma = 2 # standard deviation in frames of gaussian filter


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
seed = 3
show_unchosen_cluster = .15 #if unchosen cluster has this proportion responsibility, plot it too
do_not_overwrite = False



#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Prepare data and do cross-validation                       --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# -------------------
# Load relevant data
# -------------------
print('preparing data...')
file_location_concatenated_data = file_location + data_folder + analysis_folder + concatenated_data_name_tag + '\\' + concatenated_data_name_tag  
if os.path.isfile(file_location_concatenated_data + '_position_orientation_velocity_corrected.npy'):
    position_orientation_velocity = np.load(file_location_concatenated_data + '_position_orientation_velocity_corrected.npy')
else:
    position_orientation_velocity = np.load(file_location_concatenated_data + '_position_orientation_velocity.npy')
    print('loaded non-flip-corrected data')

pca_coeffs = np.load(file_location_concatenated_data + '_pca_coeffs.npy')

data_for_model = pca_coeffs[:,0:num_PCs_used]

#modify saved file names depending on type of clustering:
if model_sequence: 
    suffix = '_seq' + model_name_tag
else:
    suffix =  model_name_tag 


# --------------------------------
# Include velocity as a pseudo-PC
# --------------------------------
max_pc = np.max(data_for_model)
out_of_bounds = position_orientation_velocity[:,1]
trials_to_analyze_index = out_of_bounds==0 #not sheltered or out of bounds
disruptions = np.ones(len(out_of_bounds)).astype(bool)
disruptions[1:] = np.not_equal(out_of_bounds[1:],out_of_bounds[:-1])
disruptions = disruptions[trials_to_analyze_index]


if add_velocity:
    speed = position_orientation_velocity[trials_to_analyze_index,2:3]**2 + position_orientation_velocity[trials_to_analyze_index,3:4]**2 #speed
    
    speed[disruptions==1,0] = np.mean(speed[disruptions==0,0]) #remove spurious velocities
    mean_speed = np.mean(speed[:])
    std_speed = np.std(speed[:])
    speed[speed[:,0]-mean_speed > 3*std_speed,0] = mean_speed + 3*std_speed #clip spurious speeds
    
    speed_for_model = (speed-np.mean(speed)) / np.max(speed) * max_pc / vel_scaling_factor
   
    
    data_for_model = np.append(data_for_model,speed_for_model,axis=1)

# ----------------------------------------
# Include angular velocity as a pseudo-PC
# ----------------------------------------
if add_turn:
           
    head_turn_for_model = np.zeros((data_for_model.shape[0],1))
    head_direction = position_orientation_velocity[trials_to_analyze_index,4:5]
    last_head_direction = head_direction[:-1,:]
    current_head_direction = head_direction[1:,:]
    head_turn = np.min(np.concatenate( ( abs(current_head_direction - last_head_direction), abs(360-abs(current_head_direction - last_head_direction)) ),axis=1),axis=1)
    head_turn[head_turn > 180] = abs(360 - head_turn[head_turn > 180]) #algorithmic flips dont count
    head_turn[head_turn > 90] = abs(180 - head_turn[head_turn > 90])
    head_turn[head_turn > 15] = 15 #clip head turn
    head_turn[disruptions[1:]] = 0

    head_turn = (head_turn - np.mean(head_turn)) / np.max(head_turn) * max_pc / turn_scaling_factor
    
    head_turn_for_model[1:,0] = head_turn
    if filter_data_for_model: #double filter this particularly choppy feature
        head_turn_for_model = filter_features(head_turn_for_model, filter_length, sigma)
    
    data_for_model = np.append(data_for_model,head_turn_for_model,axis=1)

# -------------------------------------
# Smooth features going into the model
# -------------------------------------
if filter_data_for_model:
    data_for_model = filter_features(data_for_model, filter_length, sigma) #apply gaussian smoothing


 
    
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
np.save(file_location_concatenated_data + '_' + model_type + '_settings' + suffix + '.npy',  #save settings
        [add_velocity, True, False, add_turn, num_PCs_used, window_size, windows_to_look_at, np.max(data_for_model)])



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
print('fitting model...')
model.fit(data)   #fit model
save_file_model = file_location_concatenated_data + '_' + model_type + suffix #save model
if os.path.isfile(save_file_model) and do_not_overwrite:
    raise Exception('File already exists') 
joblib.dump(model, save_file_model)


#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Plot PCs and Clusters over Time                            --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

if plot_result:
    # -------------------------------
    # Calculate and save model output
    # -------------------------------
    calculate_and_save_model_output(data, model, num_clusters, file_location_concatenated_data, model_type, suffix, do_not_overwrite)
    
    plt.close('all')
    #reload relevant data
    components_binary = np.load(file_location_concatenated_data + '_components_binary' + suffix + '.npy')
    unchosen_components_binary = np.load(file_location_concatenated_data + '_unchosen_components_binary' + suffix + '.npy')
    unchosen_probabilities = np.load(file_location_concatenated_data + '_unchosen_probabilities' + suffix + '.npy')
    
    # set up plot
    plot_colors = ['red','deepskyblue','green','blueviolet','saddlebrown','lightpink','yellow','white']
    set_up_PC_cluster_plot(figsize=(30,10), xlim=[0,1000])
    
    # plot PCs
    data_for_model_normalized = data_for_model / np.max(data_for_model) #set max to 1 for visualization purposes
    np.save(file_location_concatenated_data+'_data_for_' + model_type + '_normalized', data_for_model_normalized)
    plt.plot(data_for_model_normalized[:,0:num_PCs_shown])
    
    # plot velocity and/or pose change
    if add_velocity:
        plt.plot(data_for_model_normalized[:,-1-add_turn], color = 'k',linewidth=2) #plot speed
        if add_turn: #plot turn speed
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
    legend_entries = create_legend(num_PCs_shown, add_velocity, True, False, add_turn)
    

