'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                  Perform PCA on wavelet-transformed 3D mouse video                             -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import linalg
from sklearn import mixture
from sklearn.model_selection import KFold
from learning_funcs import cross_validate_GMM
from sklearn.externals import joblib
import os



#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Select data file and analysis parameters                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
file_loc = 'C:\Drive\Video Analysis\data\\'
date = '05.02.2018\\'
mouse_session = '202-1a\\'
save_vid_name = 'analyze_7_3'
file_loc = file_loc + date + mouse_session + save_vid_name


# ---------------------------
# Select analysis parameters
# ---------------------------
num_clusters = 4
save_model = True
sequence_based = True
window_size = 20 #50 frames is 1 sec
windows_to_look_at = 2



# -----------------------------------
# Select cross-validation parameters
# -----------------------------------
cross_validation = False
K_range = [5] # K-fold cross-validation
seed_range = [0] # random seed used
num_clusters_range = [3,20] # range of how many clusters used in GMM
tol = .01 # how low of an error needed to end model improvement
add_velocity = True


#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Prepare data and do cross-validation                       --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# -------------------
# Load relevant data
# -------------------
probabilities = np.load(file_loc+'_probabilities.npy')
plt.close('all')


# ------------------------------------------------
# Create array of moving-window-chunked sequences
# ------------------------------------------------
box_filter = np.ones(window_size) / window_size #create boxcar filter
data_to_fit_gmm_sequence = np.zeros(probabilities.shape)
probabilities_for_gmm = data_for_gmm # use PCs
#probabilities_for_gmm[probabilities<0.01] = 0#for efficiency, set small values to 0
for cluster in range(data_to_fit_gmm_sequence.shape[1]): # apply boxcar filter
    data_to_fit_gmm_sequence[:,cluster] = np.convolve(probabilities_for_gmm[:,cluster],box_filter,mode='same')
    
responsibilities = data_to_fit_gmm_sequence
np.save(file_loc+'_responsibilities.npy',responsibilities)

data_to_concatenate_sequence = data_to_fit_gmm[window_size*windows_to_look_at:-window_size*windows_to_look_at]
if sequence_based:
    for w in range(windows_to_look_at):
    
        pre_data_to_fit_gmm_sequence = data_to_fit_gmm[window_size*(windows_to_look_at-(w+1)):-int((windows_to_look_at+w+1)*window_size),:]
        post_data_to_fit_gmm_sequence = data_to_fit_gmm[int((windows_to_look_at+w+1)*window_size):data_to_fit_gmm_sequence.shape[0]
                                                                 -window_size*(windows_to_look_at-(w+1)),:]
        data_to_concatenate_sequence = np.append(data_to_concatenate_sequence, 
                                             np.append(pre_data_to_fit_gmm_sequence, post_data_to_fit_gmm_sequence, axis=1), axis=1)   
    data_to_fit_gmm_sequence = data_to_concatenate_sequence

# --------------------
# Do cross-validation
# --------------------
if cross_validation:
    xval_scores = cross_validate_GMM(data_to_fit_gmm_sequence, [probabilities.shape[1]], K_range, seed_range, num_clusters_range, tol, 
                                     add_velocity, velocity_for_model, vel_scaling_factor)
    #xval_scores gives: % in cluster w/ high confidence; 10th percentile of confidence; score; bayesian info content
 
    

#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Generate Gaussian Mixture Model                            --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# -------------------------
# Initialize mixture model
# -------------------------
gmm_seq = mixture.GaussianMixture(n_components=num_clusters,tol=.00001,covariance_type='full',random_state=seed)


# --------------------------
# Fit and save mixture model
# --------------------------
gmm_seq.fit(data_to_fit_gmm_sequence) 

if save_model:
    save_file_model = file_loc + '_gmm_seq'
    if os.path.isfile(save_file_model) and do_not_overwrite:
        raise Exception('File already exists') 
    joblib.dump(gmm_seq, save_file_model)

#get probabilities of being in each cluster at each frame
probabilities_seq = gmm_seq.predict_proba(data_to_fit_gmm_sequence)
np.save(file_loc+'_probabilities_seq.npy',probabilities_seq)

#get which cluster is chosen at each frame, and probabilities just for that cluster
chosen_components_seq = gmm_seq.predict(data_to_fit_gmm_sequence)
np.save(file_loc+'_chosen_components_seq.npy',chosen_components_seq)
chosen_probabilities_seq = np.max(probabilities_seq,axis=1)

#get which cluster is 2nd most likely to be chosen at each frame, and probabilities just for that cluster
unchosen_probabilities_seq = probabilities_seq
for i in range(probabilities_seq.shape[0]):
    unchosen_probabilities_seq[i,chosen_components_seq[i]]=0 
unchosen_components_seq = np.argmax(unchosen_probabilities_seq,axis=1)
unchosen_probabilities_seq = np.max(unchosen_probabilities_seq,axis=1)

#get binarized version of chosen_components and unchosen_components, for later analysis
components_over_time_seq = np.zeros((data_to_fit_gmm_sequence.shape[0],num_clusters))
unchosen_components_over_time_seq = np.zeros((data_to_fit_gmm_sequence.shape[0],num_clusters))
for n in range(num_clusters):
    components_over_time_seq[:,n] = (chosen_components_seq == n)
    unchosen_components_over_time_seq[:,n] = (unchosen_components_seq == n)
np.save(file_loc+'_components_over_time_seq.npy',components_over_time_seq)
np.save(file_loc+'_unchosen_components_over_time_seq.npy',unchosen_components_over_time_seq)


#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Plot PCs and Clusters over Time                            --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# set plot parameters
plot_colors = ['red','deepskyblue','green','blueviolet','saddlebrown','lightpink','yellow','white']
plt.style.use('classic')
plt.figure(figsize=(30,10))

## plot responsibilities
#for n in range(responsibilities.shape[1]):
#    plt.plot(responsibilities[:,n],color=plot_colors[n])
    
# plot PCs
plt.plot(normalized_pca_coeffs[:,0:3])
if add_velocity:
    plt.plot(normalized_pca_coeffs[:,-2] * 2, color = 'k',linewidth=2)
    plt.plot(normalized_pca_coeffs[:,-1] * 2, color = 'gray', linestyle = '--',linewidth=2)
   
   
# plot raster of clusters above PCs
for n in range(num_clusters):
    component_frames = find(components_over_time_seq[:,n])
    plt.scatter(component_frames,np.ones(len(component_frames))*1,color=plot_colors[n],alpha=.7,marker='|',s=700)
    
    unchosen_component_frames = find(unchosen_components_over_time_seq[:,n] * (unchosen_probabilities_seq>show_unchosen_cluster))
    plt.scatter(unchosen_component_frames,np.ones(len(unchosen_component_frames))*.95,color=plot_colors[n],alpha=.4,marker='|',s=700)

# Format plot
pose_legend = np.array([])
for p in range(probabilities.shape[1]):
    pose_legend = np.append(pose_legend, ['pose ' + str(p+1)])
legend = plt.legend((pose_legend))
legend.draggable()
plt.title('Responsibilities over Time')
plt.xlabel('frame no.')
plt.ylabel('Cluster contribution')
plt.xlim([2000,5000])
plt.ylim([-1.05,1.05])
