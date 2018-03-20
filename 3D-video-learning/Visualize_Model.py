'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                           Visualize the Poses Determined by the GMM                            -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np; import cv2; from matplotlib import pyplot as plt; from mpl_toolkits.mplot3d import Axes3D
from learning_funcs import reconstruct_from_wavelet; import sys; from sklearn.externals import joblib
from learning_funcs import set_up_PC_cluster_plot, create_legend

#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Select data file and analysis parameters                 --------------------------------------
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

movie_file_loc = file_loc + date + mouse_session + 'rectified_normalized_video.avi'
file_loc = file_loc + date + mouse_session + save_vid_name

# --------------------------------
# Select visualization parameters 
# --------------------------------
model_type = 'hmm'
model_sequence = False

frame_rate = 1000
start_frame = 10
stop_frame = 30000

trajectory_pose_size = 400
show_unchosen_cluster = .15 #threshold to show 2nd-choice cluster
num_PCs_shown = 3
show_clusters = True
show_clusters_all_at_once = True



#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Load data, models, and video                    -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

#load videos
depth_vid = cv2.VideoCapture(file_loc + '_data.avi')   
#mouse_vid = cv2.VideoCapture(file_loc + '_normalized_video.avi') 
mouse_vid = cv2.VideoCapture(movie_file_loc)  

#load data
data_for_model_normalized = np.load(file_loc+'_data_for_' + model_type + '_normalized.npy')
chosen_components = np.load(file_loc+'_chosen_components.npy')
components_binary = np.load(file_loc+'_components_binary.npy')
unchosen_components_binary = np.load(file_loc+'_unchosen_components_binary.npy')
probabilities = np.load(file_loc+'_probabilities.npy')
unchosen_probabilities = np.load(file_loc+'_unchosen_probabilities.npy')

add_velocity, speed_only, add_change, num_PCs_used, window_size, windows_to_look_at, feature_max = np.load(file_loc + '_' + model_type + '_settings.npy')
add_velocity, speed_only, add_change, num_PCs_used, window_size, windows_to_look_at = np.array([add_velocity, speed_only, add_change, num_PCs_used, window_size, windows_to_look_at]).astype(int)
num_clusters = probabilities.shape[1] #load model settings

pca = joblib.load(file_loc + '_pca') #load transform info for reconstruction
relevant_ind = np.load(file_loc + '_wavelet_relevant_ind.npy')
coeff_slices = np.load(file_loc + '_wavelet_slices.npy')
level = 5 # these must be parameters taken from original wavelet transform 
discard_scale = 4

model = joblib.load(file_loc + '_' + model_type) #load model

if model_sequence: #rinse and repeat for the sequence model, if applicable
    chosen_components_seq = np.load(file_loc+'_chosen_components_seq.npy')    
    components_binary_seq = np.load(file_loc+'_components_binary_seq.npy')
    unchosen_components_binary_seq = np.load(file_loc+'_unchosen_components_binary_seq.npy')   
    probabilities_seq = np.load(file_loc+'_probabilities_seq.npy')
    unchosen_probabilities_seq = np.load(file_loc+'_unchosen_probabilities_seq.npy')
    
    num_clusters = probabilities_seq.shape[1]
    add_velocity, speed_only, add_change, num_PCs_used, window_size, windows_to_look_at, feature_max = np.load(file_loc + '_' + model_type + '_settings_seq.npy')
    add_velocity, speed_only, add_change, num_PCs_used, window_size, windows_to_look_at = np.array([add_velocity, speed_only, add_change, num_PCs_used, window_size, windows_to_look_at]).astype(int)
    
    model = joblib.load(file_loc + '_' + model_type + '_seq')
    



#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Plot Mean Poses                                  -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------
# Get mean and variance of clusters
# ---------------------------------
mean_features_model = model.means_
if model_type == 'hmm':
    var_features_model = model.covars_
elif model_type == 'gmm':
    var_features_model = model.covariances_
    

# ------------------------------------------------
# Display mean trajectory or pose for each cluster
# ------------------------------------------------
# Set up colors
colors = [[0,0,255],[169,118,14],[10,205,10],[160,0,120],[0,80,120],[170,120,220],[0,140,140],[100,100,100]]*3
plot_colors = ['red','deepskyblue','green','blueviolet','orange','lightpink','yellow','white']

color_array = np.zeros((trajectory_pose_size,trajectory_pose_size,3,len(colors))) #create coloring arrays
for c in range(len(colors)):
    for i in range(3): #B, G, R
        color_array[:,:,i,c] = np.ones((trajectory_pose_size,trajectory_pose_size)) * colors[c][i] / sum(colors[c])
        
        
# Get the number of features used in model
features_used = int(mean_features_model.shape[1] / (2*windows_to_look_at + 1)) 

trajectory_position = [windows_to_look_at] #get indices to put the various clusters in the right order, below
offset = 1
for i in range(2*windows_to_look_at):
    trajectory_position.append(windows_to_look_at + offset * (i%2==1) - + offset * (i%2==0))
    if (i%2==1):
        offset+=1
    
# Plot the mean trajectory or pose:
for n in range(num_clusters):
    trajectory = np.zeros((trajectory_pose_size,trajectory_pose_size*(2*windows_to_look_at + 1),3)).astype(np.uint8) #initialize mean trajectory image
    for t in range(2*windows_to_look_at + 1):
        # Get the mean features for that pose
        mean_features = mean_features_model[n,t*features_used:(t+1)*features_used] 
        
        # Reconstruct wavelet-transformed data from the PCs
        mean_wavelet_relevant_features = pca.inverse_transform(np.append(mean_features[0:num_PCs_used],np.zeros(12-num_PCs_used)))
        mean_wavelet = np.zeros(39*39)
        mean_wavelet[relevant_ind] = mean_wavelet_relevant_features
        mean_wavelet_array = np.reshape(mean_wavelet,(39,39))
         
        # Reconstruct image in pixel space from wavelet-transformed reconstruction
        reconstruction_from_wavelet  = reconstruct_from_wavelet(mean_wavelet_array,coeff_slices, level, discard_scale)
        reconstruction_image= cv2.resize(abs(reconstruction_from_wavelet).astype(np.uint8),(trajectory_pose_size,trajectory_pose_size))
        reconstruction_image= cv2.cvtColor(reconstruction_image, cv2.COLOR_GRAY2BGR)
        reconstruction_image = (reconstruction_image * np.squeeze(color_array[:,:,:,n])).astype(uint8)
        
        #rotate image
        M = cv2.getRotationMatrix2D((int(trajectory_pose_size/2),int(trajectory_pose_size/2)),-90,1)
        reconstruction_image = cv2.warpAffine(reconstruction_image,M,(trajectory_pose_size,trajectory_pose_size)) 
    
        # Also display an arrow indicating the mean velocity of that cluster
        if add_velocity:
            velocity_of_current_epoch = mean_features_model[:,(t+1)*features_used-2-add_change+speed_only]
            
            #if velocity includes negative values for whatever reason, make only positive by subtracting the lowest velocity value
            min_velocity = np.min(mean_features_model[:,np.arange(features_used-2-add_change+speed_only,mean_features_model.shape[1] ,features_used)])
            max_velocity = np.max(mean_features_model[:,np.arange(features_used-2-add_change+speed_only,mean_features_model.shape[1] ,features_used)])
            if min_velocity < 0:
                velocity_of_current_epoch = velocity_of_current_epoch - min_velocity
                max_velocity = max_velocity - min_velocity
                
            arrow_height = velocity_of_current_epoch[n] / max_velocity
            
            if not(speed_only) or add_change:
                arrow_sideness = ((mean_features_array[(t+1)*features_used-1-add_change+speed_only,n] / np.max(mean_features_model[:,(t+1)*features_used-2+add_change])) + 1) / 2
            else:
                arrow_sideness = 0
            cv2.arrowedLine(reconstruction_image,(10, trajectory_pose_size-10),(10+int(20*arrow_height), trajectory_pose_size - 10 - int(20*arrow_sideness)),(250,250,250),thickness=2)
#        
        # Display mean pose
        title = 'trajectory ' + str(n+1)
        trajectory[:,trajectory_pose_size*(t):trajectory_pose_size*(t+1),:] = reconstruction_image
        cv2.imshow(title,trajectory)
        if cv2.waitKey(int(100)) & 0xFF == ord('q'):
            break



#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                  Plot PCs, Clusters, and Video over time                    -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------
# Set up normalized and data videos, and component data
# -----------------------------------------------------
num_frames = int(depth_vid.get(cv2.CAP_PROP_FRAME_COUNT))
depth_vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
mouse_vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)

if model_sequence:
    components_binary_to_display = components_binary_seq
    unchosen_components_binary_to_display = unchosen_components_binary_seq
    unchosen_probabilities_to_display = unchosen_probabilities_seq
    chosen_components_to_display = chosen_components_seq
else:
    components_binary_to_display = components_binary
    unchosen_components_binary_to_display = unchosen_components_binary
    unchosen_probabilities_to_display = unchosen_probabilities
    chosen_components_to_display = chosen_components

# -------------------------
# Set up feature-space plot
# -------------------------
plt.close('all' )

if show_clusters:
    fig = plt.figure('PC trajectory',figsize=(10,10))
    ax_3D = fig.add_subplot(111, projection='3d')
    mean_features_model_normalized = mean_features_model / feature_max
    
    for n in range(num_clusters):
        if show_clusters_all_at_once:
            component_frames = find(components_binary_to_display[:,n])
            ax_3D.scatter(data_for_model_normalized[component_frames,0], data_for_model_normalized[component_frames,1], 
                       data_for_model_normalized[component_frames,2],color=plot_colors[n])
        ax_3D.scatter(mean_features_model_normalized[n,0],mean_features_model_normalized[n,1],
                   mean_features_model_normalized[n,2],color=plot_colors[n],s=5000, alpha = .3)
    
    ax_3D.set_xlabel('PC1'); ax_3D.set_ylabel('PC2'); ax_3D.set_zlabel('PC3')
    ax_3D.set_title('Clusters in PC space')


# ----------------------------
# Set up moving features plot
# ----------------------------
rolling_fig = set_up_PC_cluster_plot(figsize=(20,7), xlim=[1500,2500])
ax_2D = rolling_fig.add_subplot(111)

# create coloring array
color_array = np.zeros((450,450,3,len(colors)))
for c in range(len(colors)):
    for i in range(3): #B, G, R
        color_array[:,:,i,c] = np.ones((450,450)) * colors[c][i] / sum(colors[c])
        
   
# Plot the corresponding poses along y = 0 for sequence analysis
if model_sequence:
    for n in range(probabilities.shape[1]):
        component_frames = find(components_binary[:,n])
        ax_2D.scatter(component_frames,np.ones(len(component_frames))*0,color=plot_colors[n],alpha=.5,marker='|',s=700)


# Now plot the components of the chosen model (pose or sequence)
for n in range(num_clusters):
    component_frames = find(components_binary_to_display[:,n])
    ax_2D.scatter(component_frames,np.ones(len(component_frames))*1,color=plot_colors[n],alpha=.7,marker='|',s=700)
    
    unchosen_component_frames = find(unchosen_components_binary_to_display[:,n] * (unchosen_probabilities_to_display>show_unchosen_cluster))
    ax_2D.scatter(unchosen_component_frames,np.ones(len(unchosen_component_frames))*.9,color=plot_colors[n],alpha=.4,marker='|',s=700)

# plot the desired number of pc coefficients, varying in time
ax_2D.plot(data_for_model_normalized[:,0:num_PCs_shown],linewidth=2)

# plot the velocity
if add_velocity:
    ax_2D.plot(data_for_model_normalized[:,-2-add_change+speed_only], color = 'k',linewidth=2) #plot speed
    if not(speed_only) or add_change: #plot turn speed or, if available, pose change
        ax_2D.plot(data_for_model_normalized[:,-1], color = 'gray', linestyle = '--',linewidth=2)

# Create legend
legend_entries = create_legend(num_PCs_shown, add_velocity, speed_only, add_change)

# Create line indicating which frame we're looking at
center_line = ax_2D.plot([0,0],[-2,2],color = 'gray', linestyle = '--')



# ------------------------------------------------------------
# Show behaviour and 3D videos, with selected pose also shown 
# ------------------------------------------------------------
i = start_frame - window_size*windows_to_look_at + 1
while True:
    ret1, frame1 = depth_vid.read() 
    ret2, frame2 = mouse_vid.read()
    
    if ret1 and ret2:
        # Grab frame number, which cluster is active, and its corresponding color
        frame_num = int(depth_vid.get(cv2.CAP_PROP_POS_FRAMES))

        chosen_component = chosen_components_to_display[i]
        chosen_color = colors[chosen_component]       
        
        # Grab the images to be displayed
        depth_frame = frame1[:,:,0]
        mouse_frame = frame2[:,:,0]
        
        # Resize and recolor images
        depth_frame = cv2.resize(depth_frame,(450,450))
        depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
        mouse_frame = cv2.cvtColor(mouse_frame, cv2.COLOR_GRAY2BGR)        
        depth_frame = (depth_frame * np.squeeze(color_array[:,:,:,chosen_component])).astype(uint8)

        # Add colored circle and frame number to behaviour image
        cv2.circle(mouse_frame,(50,50),radius=25, color=chosen_color,thickness=50)
        cv2.putText(mouse_frame,str(i),(50,400),0,1,255)
        
        # Display Images
        cv2.imshow('depth',depth_frame)
        cv2.imshow('behaviour',mouse_frame)
        
        # Move the PC / clusters plot to be centered at the current frame
        center_line.pop(0).remove()
        center_line = ax_2D.plot([i,i],[-2,2],color = 'gray', linestyle = '--')
        ax_2D.set_xlim([i-500,i+500])
        plt.pause(0.01)
        
        #add to trajectory plot
        if show_clusters and not show_clusters_all_at_once:
            ax_3D.scatter(data_for_model_normalized[i,0], data_for_model_normalized[i,1], 
                   data_for_model_normalized[i,-1],color=plot_colors[chosen_component],s=100,alpha=.5)
        
        # update current frame index
        i = frame_num - window_size*windows_to_look_at + 1
        
        # stop video when donw
        if (frame_num)%500==0:
            print(str(frame_num) + ' out of ' + str(num_frames) + ' frames complete')   
        if depth_vid.get(cv2.CAP_PROP_POS_FRAMES) >= stop_frame or depth_vid.get(cv2.CAP_PROP_POS_FRAMES) >= num_frames:
            break 
        if cv2.waitKey(int(1000/frame_rate)) & 0xFF == ord('q'):
            break
    else:
        print('Problem with Video Playback...')
        cv2.waitKey(int(500))
      
depth_vid.release()
