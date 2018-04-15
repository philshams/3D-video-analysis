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

            #file_name = '3Dtest_secondmouse0.avi' #
            #
            #file_loc = 'C:\Drive\Video Analysis\data\\'
            #date = '14.02.2018_zina\\' #
            #mouse_session = 'twomouse\\'  #
            #
            #save_vid_name = 'analyze'

file_loc = file_loc + date + mouse_session + save_vid_name


# ---------------------------
# Select analysis parameters
# ---------------------------
add_velocity = True
vel_scaling_factor = 2 #scale down velocity's importance relative to PC1
save_model = True

num_PCs_used = 4
num_clusters = 4
filter_pcs = True
filter_length = 5 # total width of gaussian smoothing
sigma = 2 #standard deviation in frames of gaussian filter
seed = 1
show_unchosen_cluster = .15 #if unchosen cluster has this proportion responsibility, plot it too

do_not_overwrite = False
display_sub_clusters = False

# -----------------------------------
# Select cross-validation parameters
# -----------------------------------
cross_validation = False
num_PCs_range = [6] # how many PCs are used
K_range = [10] # K-fold cross-validation
seed_range = [0] # random seed used
num_clusters_range = [3,8] # range of how many clusters used in GMM
tol = .001 # how low of an error needed to end model improvement




#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Prepare data and do cross-validation                       --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# -------------------
# Load relevant data
# -------------------
pca_coeffs = np.load(file_loc + '_pca_coeffs.npy')
velocity = np.load(file_loc + '_velocity.npy')
#plt.close('all')

# --------------------------------
# Include velocity as a pseudo-PC
# --------------------------------
if add_velocity:
    max_vel = np.max(abs(velocity[:,0:2]))
    max_pc = np.max(pca_coeffs[:,0])
    velocity_for_model = velocity[:,0:2] / max_vel * max_pc / vel_scaling_factor
    velocity_for_model[:,1] = np.abs(velocity_for_model[:,1]) - np.mean(np.abs(velocity_for_model[:,1]))
else:
    velocity_for_model = 0
    vel_scaling_factor = 0

# --------------------
# Do cross-validation
# --------------------
if cross_validation:
    xval_scores = cross_validate_GMM(pca_coeffs, num_PCs_range, K_range, seed_range, num_clusters_range, tol, 
                                     add_velocity, velocity_for_model, vel_scaling_factor)
    #xval_scores gives: % in cluster w/ high confidence; 10th percentile of confidence; score; bayesian info content
   

#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Generate Gaussian Mixture Model                            --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# -------------------------
# Initialize mixture model
# -------------------------
gmm = mixture.GaussianMixture(n_components=num_clusters,tol=.00001,covariance_type='full',random_state=seed)
data_to_generate_gmm = pca_coeffs[:,0:num_PCs_used]

if add_velocity:  # append velocity as pseudo-PC
    data_for_gmm = np.append(data_to_generate_gmm,velocity_for_model,axis=1)


# -----------------------------------
# Smooth features going into the GMM
# -----------------------------------
if filter_pcs:
    gauss_filter = np.exp(-np.arange(-filter_length,filter_length+1)**2/sigma**2)  #create filter
    gauss_filter = gauss_filter / sum(gauss_filter) # normalize filter

    data_to_fit_gmm = np.zeros(data_for_gmm.shape)
    for pc in range(data_for_gmm.shape[1]): # apply filter
        data_to_fit_gmm[:,pc] = np.convolve(data_for_gmm[:,pc],gauss_filter,mode='same')
else:
    data_to_fit_gmm = data_to_generate_gmm


# --------------------------
# Fit and save mixture model
# --------------------------
gmm.fit(data_to_fit_gmm) 
if save_model:
    save_file_model = file_loc + '_gmm'
    if os.path.isfile(save_file_model) and do_not_overwrite:
        raise Exception('File already exists') 
    joblib.dump(gmm, save_file_model)

#get probabilities of being in each cluster at each frame
probabilities = gmm.predict_proba(data_to_fit_gmm)
np.save(file_loc+'_probabilities.npy',probabilities)

#get which cluster is chosen at each frame, and probabilities just for that cluster
chosen_components = gmm.predict(data_to_fit_gmm)
np.save(file_loc+'_chosen_components.npy',chosen_components)
chosen_probabilities = np.max(probabilities,axis=1)

#get which cluster is 2nd most likely to be chosen at each frame, and probabilities just for that cluster
unchosen_probabilities = probabilities
for i in range(probabilities.shape[0]):
    unchosen_probabilities[i,chosen_components[i]]=0 
unchosen_components = np.argmax(unchosen_probabilities,axis=1)
unchosen_probabilities = np.max(unchosen_probabilities,axis=1)

#get binarized version of chosen_components and unchosen_components, for later analysis
components_over_time = np.zeros((data_to_fit_gmm.shape[0],num_clusters))
unchosen_components_over_time = np.zeros((data_to_fit_gmm.shape[0],num_clusters))
for n in range(num_clusters):
    components_over_time[:,n] = (chosen_components == n)
    unchosen_components_over_time[:,n] = (unchosen_components == n)
np.save(file_loc+'_components_over_time.npy',components_over_time)
np.save(file_loc+'_unchosen_components_over_time.npy',unchosen_components_over_time)


#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Plot PCs and Clusters over Time                            --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# normalize pca coefficients to max = 1
data_to_fit_gmm_normalized = data_to_fit_gmm / np.max(data_to_fit_gmm)
np.save(file_loc+'_data_for_gmm_normalized.npy',data_to_fit_gmm_normalized)

# set plot parameters
plot_colors = ['red','deepskyblue','green','blueviolet','saddlebrown','lightpink','yellow','white']
plt.style.use('classic')
plt.figure(figsize=(30,10))

# plot PCs
plt.plot(data_to_fit_gmm_normalized[:,0:3])
if add_velocity:
    plt.plot(data_to_fit_gmm_normalized[:,-2] * 2, color = 'k',linewidth=2)
    plt.plot(data_to_fit_gmm_normalized[:,-1] * 2, color = 'gray', linestyle = '--',linewidth=2)
    
# plot raster of clusters above PCs
for n in range(num_clusters):
    component_frames = find(components_over_time[:,n])
    plt.scatter(component_frames,np.ones(len(component_frames))*1,color=plot_colors[n],alpha=.7,marker='|',s=700)
    
    unchosen_component_frames = find(unchosen_components_over_time[:,n] * (unchosen_probabilities>show_unchosen_cluster))
    plt.scatter(unchosen_component_frames,np.ones(len(unchosen_component_frames))*.9,color=plot_colors[n],alpha=1,marker='|',s=700)

# Format plot
legend = plt.legend(('PC1','PC2','PC3','direct velocity','ortho velocity'))
legend.draggable()
plt.title('Principal Components over Time')
plt.xlabel('frame no.')
plt.ylabel('PC amplitude')
plt.xlim([1500,3500])
plt.ylim([-1,1.05])



#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Get distribution over sub-clusters                         --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

if display_sub_clusters:
    # calculate secondary clusters based on threshold set above
    dual_components = (chosen_components+1)*10+(unchosen_components+1)*(unchosen_probabilities>show_unchosen_cluster)
    
    # Create labels for histogram
    labels = np.array([])
    for n in range(num_clusters):
        labels = np.append(labels,(10*(n+1) + arange(num_clusters+1)))
    
    # Plot data
    plt.figure(figsize=(40,7))
    for n in range(num_clusters):
        plt.hist(dual_components[(dual_components>=(n+1)*10) * (dual_components<(n+2)*10)],label=labels.astype(str),bins = np.arange(10,61),density = True, color = plot_colors[n])
    
    # Format plot
    plt.xlim([10,(num_clusters+1)*10])
    plt.ylim([0,.8])
    plt.xticks(labels)
    plt.title('Distribution of clusters and sub-clusters')
    




#%% Do autoregression in 1-D (cluster) or 9-D (PC) space and use prediction correctness to generate changepoints


#%% Identify changepoints over different temporal scales, using different temporal filters









