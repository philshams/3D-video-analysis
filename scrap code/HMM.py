'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                  Perform PCA on wavelet-transformed 3D mouse video                             -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from learning_funcs import import_all, cross_validate_GMM


#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Select data file and analysis parameters                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
file_loc = 'C:\Drive\Video Analysis\data\\'
date = '05.02.2018\\'
#date = '14.02.2018_zina\\' #
mouse_session = '202-1a\\'
#mouse_session = 'twomouse\\'  #
save_vid_name = 'analyze_7_3'
#save_vid_name = 'analyze'

file_loc = file_loc + date + mouse_session + save_vid_name


# ---------------------------
# Select analysis parameters
# ---------------------------
# Primary Settings
num_clusters = 4 #number of poses
num_PCs_used = 4 #number of PCs used as features
add_velocity = True #include velocity as a pseudo-PC
vel_scaling_factor = 2 #scale down velocity's importance relative to PC1
add_change = False #include that change in PC-space since last frame as a pseudo-PC

# Smoothing Settings
filter_pcs = True
filter_length = 5 # total width of gaussian smoothing
sigma = 2 #standard deviation in frames of gaussian filter

# Cross-Validation Settings
cross_validation = False
num_PCs_range = [6] # how many PCs are used
K_range = [10] # K-fold cross-validation
seed_range = [0] # random seed used
num_clusters_range = [3,8] # range of how many clusters used in hmm
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
pca_coeffs = np.load(file_loc + '_pca_coeffs.npy')
velocity = np.load(file_loc + '_velocity.npy')
data_for_hmm = pca_coeffs[:,0:num_PCs_used]
plt.close('all')

# --------------------------------
# Include velocity as a pseudo-PC
# --------------------------------
if add_velocity:
    if add_change: #use either both add_change and speed_only or neither
        speed_only = True
    else:
        speed_only = False
    max_vel = np.max(abs(velocity[:,0:2]))
    max_pc = np.max(pca_coeffs[:,0])
    
    if not speed_only: #get forward and turn velocity
        velocity_for_model = np.zeros((data_for_hmm.shape[0],2))
        velocity_for_model[:,:] = velocity[:,0:2] / max_vel * max_pc / vel_scaling_factor
        velocity_for_model[:,1] = np.abs(velocity_for_model[:,1]) - np.mean(np.abs(velocity_for_model[:,1]))
        
    if speed_only: #just get velocity magnitude (speed)
        speed_for_model = np.zeros((data_for_hmm.shape[0],1))
        speed_for_model[:,0] = np.nan_to_num(np.sqrt(velocity[:,0]**2 + velocity[:,1]**1))
        velocity_for_model = speed_for_model - np.mean(speed_for_model)
        max_speed = np.max(abs(velocity_for_model))
        velocity_for_model = velocity_for_model / max_speed * max_pc / vel_scaling_factor
    #append the appropriate velocity array to PCs    
    data_for_hmm = np.append(data_for_hmm,velocity_for_model,axis=1)
else:
    velocity_for_model = 0
    vel_scaling_factor = 0
    
# --------------------------------------
# Include change in pose as a pseudo-PC
# --------------------------------------
if add_change:
    ch_ch_ch_ch_changes = np.zeros((data_for_hmm.shape[0],1))
    ch_ch_ch_ch_changes[:,0] = np.append(0,  #just the norm of the difference in PC space of consecutive frames
        np.linalg.norm(data_for_hmm[1:,0:num_PCs_used] - data_for_hmm[:-1,0:num_PCs_used],axis=1))
    ch_ch_ch_ch_changes = ch_ch_ch_ch_changes - np.mean(ch_ch_ch_ch_changes)
    max_change = np.max(ch_ch_ch_ch_changes)
    ch_ch_ch_ch_changes = ch_ch_ch_ch_changes / max_change * max_pc / vel_scaling_factor
    #append the change array to PCs and velocity
    data_for_hmm = np.append(data_for_hmm,ch_ch_ch_ch_changes,axis=1)
    
# -----------------------------------
# Smooth features going into the hmm
# -----------------------------------
if filter_pcs:
    gauss_filter = np.exp(-np.arange(-filter_length,filter_length+1)**2/sigma**2)  #create filter
    gauss_filter = gauss_filter / sum(gauss_filter) # normalize filter

    data_to_filter_hmm = np.zeros(data_for_hmm.shape)
    for pc in range(data_for_hmm.shape[1]): # apply filter to each feature
        data_to_filter_hmm[:,pc] = np.convolve(data_for_hmm[:,pc],gauss_filter,mode='same')
    data_for_hmm = data_to_filter_hmm


# --------------------
# Do cross-validation
# --------------------
if cross_validation:
    xval_scores = cross_validate_hmm(data_for_hmm, num_PCs_range, K_range, seed_range, num_clusters_range, tol, 
                                     add_velocity, velocity_for_model, vel_scaling_factor)
    #xval_scores gives: % in cluster w/ high confidence; 10th percentile of confidence; score; bayesian info content
   

#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Generate Gaussian Mixture Model                            --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------
# Initialize and fit mixture model
# --------------------------------
hmm = hmm_model.GaussianHMM(n_components=num_clusters, covariance_type="full",algorithm='viterbi',tol=.001,random_state=seed)
hmm.fit(data_to_fit_hmm) 

# --------------------------
# Save model output
# --------------------------
#save model itself
save_file_model = file_loc + '_hmm'
if os.path.isfile(save_file_model) and do_not_overwrite:
    raise Exception('File already exists') 
joblib.dump(hmm, save_file_model)

#get probabilities of being in each cluster at each frame
probabilities = hmm.predict_proba(data_to_fit_hmm)
np.save(file_loc+'_probabilities.npy',probabilities)

#get which cluster is chosen at each frame, and probabilities just for that cluster
chosen_components = hmm.predict(data_to_fit_hmm)
np.save(file_loc+'_chosen_components.npy',chosen_components)
chosen_probabilities = np.max(probabilities,axis=1)

#get which cluster is 2nd most likely to be chosen at each frame, and probabilities just for that cluster
unchosen_probabilities = probabilities
for i in range(probabilities.shape[0]):
    unchosen_probabilities[i,chosen_components[i]]=0 
unchosen_components = np.argmax(unchosen_probabilities,axis=1)
unchosen_probabilities = np.max(unchosen_probabilities,axis=1)

#get binarized version of chosen_components and unchosen_components, for later analysis
components_over_time = np.zeros((data_to_fit_hmm.shape[0],num_clusters))
unchosen_components_over_time = np.zeros((data_to_fit_hmm.shape[0],num_clusters))
for n in range(num_clusters):
    components_over_time[:,n] = (chosen_components == n)
    unchosen_components_over_time[:,n] = (unchosen_components == n)
np.save(file_loc+'_components_over_time.npy',components_over_time)
np.save(file_loc+'_unchosen_components_over_time.npy',unchosen_components_over_time)

#save settings
np.save(file_loc + '_hmm_settings.npy', [add_velocity, speed_only, add_change, num_PCs_used])


#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Plot PCs and Clusters over Time                            --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# normalize data_for_hmm to max = 1
data_for_hmm_normalized = data_for_hmm / np.max(data_for_hmm)
np.save(file_loc+'_data_for_hmm_normalized.npy',data_for_hmm_normalized)

if plot_result:
    plt.close('all')
    
    # set plot parameters
    plot_colors = ['red','deepskyblue','green','blueviolet','saddlebrown','lightpink','yellow','white']
    plt.style.use('classic')
    plt.figure('PCs and clusters',figsize=(30,10))
    plt.title('Principal Components and Poses over Time')
    plt.xlabel('frame no.')
    plt.ylabel('PC amplitude')
    plt.xlim([1500,2500])
    plt.ylim([-1,1.05])
    
    # plot PCs
    plt.plot(data_for_hmm_normalized[:,0:3])
    
    # plot velocity and/or pose change
    if add_velocity:
        plt.plot(data_to_fit_model_normalized[:,-2-add_change+speed_only], color = 'k',linewidth=2) #plot speed
        if not(speed_only) or add_change: #plot turn speed or, if available, pose change
            plt.plot(data_to_fit_model_normalized[:,-1], color = 'gray', linestyle = '--',linewidth=2)

    # plot raster of clusters above PCs
    for n in range(num_clusters):
        #plot chosen pose:
        component_frames = find(components_over_time[:,n]) 
        plt.scatter(component_frames,np.ones(len(component_frames))*1,color=plot_colors[n],alpha=.7,marker='|',s=700)
        
        #plot 2nd-place pose:
        unchosen_component_frames = find(unchosen_components_over_time[:,n] * (unchosen_probabilities>show_unchosen_cluster)) 
        plt.scatter(unchosen_component_frames,np.ones(len(unchosen_component_frames))*.9,color=plot_colors[n],alpha=1,marker='|',s=700)
    
    # Create legend
    legend_entries = []
    for pc in range(num_PCs_shown):
        legend_entries.append('PC' + str(pc+1))
    if add_velocity:
        legend_entries.append('speed')
        if not(speed_only):
            legend_entries.append('turn speed')
    if add_change:
        legend_entries.append('change in pose')
    legend = plt.legend((legend_entries))
    





