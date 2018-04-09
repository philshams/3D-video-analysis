'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                           Visualize the Poses Determined by the Model                            -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np; import cv2; from matplotlib import pyplot as plt; from mpl_toolkits.mplot3d import Axes3D; import os
from learning_funcs import reconstruct_from_wavelet; import sys; from sklearn.externals import joblib
from learning_funcs import set_up_PC_cluster_plot, create_legend

#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Select data file and analysis parameters                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
# Select model directory
file_location = 'C:\Drive\Video Analysis\data\\'
data_folder = 'baseline_analysis\\'
analysis_folder = 'together_for_model\\'

concatenated_data_name_tag = 'analyze' 


# Select session directory and video
session_name_tag = 'normal_1_0'
session_video_folder = 'C:\\Drive\\Video Analysis\\data\\baseline_analysis\\27.02.2018\\205_1a\\'
session_video = session_video_folder + 'Chronic_Mantis_stim-default-996386-video-0.avi'

# --------------------------------
# Select visualization parameters 
# --------------------------------
model_type = 'hmm'
model_sequence = False

frame_rate = 1000
start_frame = 5000
stop_frame = 40000

show_clusters = False
show_clusters_all_at_once = False
show_shelter = True
show_PC_plot = True

only_see_component = 3
only_see = False

model_name_tag = ''


#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Load data, models, and video                    -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

#load videos
file_location_concatenated_data = file_location + data_folder + analysis_folder + concatenated_data_name_tag + '\\' + concatenated_data_name_tag  
file_location_data = file_location + data_folder + analysis_folder + session_name_tag + '\\' + session_name_tag
session_name_tags = np.load(file_location_concatenated_data + '_session_name_tags.npy')
print(concatenated_data_name_tag)
print(session_name_tag)  
session_num = find(session_name_tags==session_name_tag)[0]
mouse_vid = cv2.VideoCapture(session_video) 


#load data
data_for_model_normalized = np.load(file_location_concatenated_data +'_data_for_' + model_type + '_normalized.npy')
chosen_components = np.load(file_location_concatenated_data+'_chosen_components' + str(model_name_tag) + '.npy')
components_binary = np.load(file_location_concatenated_data+'_components_binary' + str(model_name_tag) + '.npy')
unchosen_components_binary = np.load(file_location_concatenated_data+'_unchosen_components_binary' + str(model_name_tag) + '.npy')
probabilities = np.load(file_location_concatenated_data+'_probabilities' + str(model_name_tag) + '.npy')
unchosen_probabilities = np.load(file_location_concatenated_data+'_unchosen_probabilities' + str(model_name_tag) + '.npy')

add_velocity, speed_only, add_change, add_turn, num_PCs_used, window_size, windows_to_look_at, feature_max = np.load(file_location_concatenated_data + '_' + model_type + '_settings.npy')
add_velocity, speed_only, add_change, add_turn, num_PCs_used, window_size, windows_to_look_at = np.array([add_velocity, speed_only, add_change, add_turn, num_PCs_used, window_size, windows_to_look_at]).astype(int)
num_clusters = probabilities.shape[1] #load model settings

pca = joblib.load(file_location_concatenated_data + '_pca') #load transform info for reconstruction
relevant_ind = np.load(file_location_concatenated_data + '_wavelet_relevant_ind_PCA.npy')
coeff_slices = np.load(file_location_concatenated_data + '_wavelet_slices.npy')
level = 5 # these must be parameters taken from original wavelet transform 
discard_scale = 4

if os.path.isfile(file_location_concatenated_data + '_position_orientation_velocity_corrected.npy'):
    position_orientation_velocity = np.load(file_location_concatenated_data + '_position_orientation_velocity_corrected.npy')
else:
    position_orientation_velocity = np.load(file_location_concatenated_data + '_position_orientation_velocity.npy')
    print('loaded non-flip-corrected data')
    
session_index = position_orientation_velocity[:,7] == session_num
out_of_bounds = position_orientation_velocity[session_index,1]
frames = position_orientation_velocity[session_index,0]
frames = frames[out_of_bounds==0]

head_dir = position_orientation_velocity[:,4]
vel_forward = position_orientation_velocity[:,2]
vel_ortho = position_orientation_velocity[:,3]
pos_x = position_orientation_velocity[:,5]
pos_y = position_orientation_velocity[:,6]
disruptions = np.ones(len(out_of_bounds)).astype(bool)
disruptions[1:] = np.not_equal(out_of_bounds[1:],out_of_bounds[:-1])


if show_shelter:
    shelter = cv2.imread('C:\\Drive\\Video Analysis\\data\\calibration_images\\shelter\\shelter.png')
    shelter = cv2.resize(shelter,(450,450))
    
model = joblib.load(file_location_concatenated_data + '_' + model_type + str(model_name_tag)) #load model

if model_sequence: #rinse and repeat for the sequence model, if applicable
    chosen_components_seq = np.load(file_location_concatenated_data+'_chosen_components_seq.npy')    
    components_binary_seq = np.load(file_location_concatenated_data+'_components_binary_seq.npy')
    unchosen_components_binary_seq = np.load(file_location_concatenated_data+'_unchosen_components_binary_seq.npy')   
    probabilities_seq = np.load(file_location_concatenated_data+'_probabilities_seq.npy')
    unchosen_probabilities_seq = np.load(file_location_concatenated_data+'_unchosen_probabilities_seq.npy')
    
    num_clusters = probabilities_seq.shape[1]
    add_velocity, speed_only, add_change, add_turn, num_PCs_used, window_size, windows_to_look_at, feature_max = np.load(file_location_concatenated_data + '_' + model_type + '_settings_seq.npy')
    add_velocity, speed_only, add_change, add_turn, num_PCs_used, window_size, windows_to_look_at = np.array([add_velocity, speed_only, add_change, add_turn, num_PCs_used, window_size, windows_to_look_at]).astype(int)
    
    model = joblib.load(file_location_concatenated_data + '_' + model_type + '_seq')
    



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
trajectory_pose_size = 400

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
for n in [2]:
    #reencode and save video
    file_location_save = file_loc = 'C:\Drive\Video Analysis\data\\calibration_images\\shelter\\'
    fourcc = cv2.VideoWriter_fourcc(*'XVID') #LJPG for lossless, XVID or MJPG works for compressed
    rotator = cv2.VideoWriter(file_location_save + str(n) +'.avi', fourcc , 40, (trajectory_pose_size,trajectory_pose_size)) 

    for i in range(10):
        for angle in range(360):
            trajectory = np.zeros((trajectory_pose_size,trajectory_pose_size*(2*windows_to_look_at + 1),3)).astype(np.uint8) #initialize mean trajectory image
            for t in range(2*windows_to_look_at + 1):
                # Get the mean features for that pose
                mean_features = mean_features_model[n,t*features_used:(t+1)*features_used] 
                
                # Reconstruct wavelet-transformed data from the PCs
                mean_wavelet_relevant_features = pca.inverse_transform(np.append(mean_features[0:num_PCs_used],np.zeros(10-num_PCs_used)))
                mean_wavelet = np.zeros(39*39)
                mean_wavelet[relevant_ind] = mean_wavelet_relevant_features
                mean_wavelet_array = np.reshape(mean_wavelet,(39,39))
                 
                # Reconstruct image in pixel space from wavelet-transformed reconstruction
                reconstruction_from_wavelet  = reconstruct_from_wavelet(mean_wavelet_array,coeff_slices, level, discard_scale)
                reconstruction_image= cv2.resize(abs(reconstruction_from_wavelet).astype(np.uint8),(trajectory_pose_size,trajectory_pose_size))
                reconstruction_image= cv2.cvtColor(reconstruction_image, cv2.COLOR_GRAY2BGR)
                reconstruction_image = (reconstruction_image * np.squeeze(color_array[:,:,:,n])).astype(uint8)
                
                #rotate image
                M = cv2.getRotationMatrix2D((int(trajectory_pose_size/2),int(trajectory_pose_size/2)),-90-angle,1)
                reconstruction_image = cv2.warpAffine(reconstruction_image,M,(trajectory_pose_size,trajectory_pose_size)) 
            
                # Also display an arrow indicating the mean velocity of that cluster
                if add_velocity:
                    velocity_of_current_epoch = mean_features_model[:,(t+1)*features_used-1-add_turn]
                    #if velocity includes negative values for whatever reason, make only positive by subtracting the lowest velocity value
                    min_velocity = np.min(mean_features_model[:,np.arange(features_used-1-add_turn,mean_features_model.shape[1] ,features_used)])
                    max_velocity = np.max(mean_features_model[:,np.arange(features_used-1-add_turn,mean_features_model.shape[1] ,features_used)])
                    if min_velocity < 0:
                        velocity_of_current_epoch = velocity_of_current_epoch - min_velocity
                        max_velocity = max_velocity - min_velocity
                    arrow_speed = velocity_of_current_epoch[n] / max_velocity
                    
                    if add_turn:
                        turn_of_current_epoch = mean_features_model[:,(t+1)*features_used-1]
                        #if velocity includes negative values for whatever reason, make only positive by subtracting the lowest velocity value
                        min_turn = np.min(mean_features_model[:,np.arange(features_used-1,mean_features_model.shape[1] ,features_used)])
                        max_turn = np.max(mean_features_model[:,np.arange(features_used-1,mean_features_model.shape[1] ,features_used)])
                        if min_turn < 0:
                            turn_of_current_epoch = turn_of_current_epoch - min_turn
                            max_turn = max_turn - min_turn
                        arrow_turn = turn_of_current_epoch[n] / max_turn
        #                arrow_turn = (mean_features_model[n,(t+1)*features_used-1] / np.max(mean_features_model[:,(t+1)*features_used-1]))
        
                        
                    else:
                        arrow_turn = 0
                    cv2.arrowedLine(reconstruction_image,(10, trajectory_pose_size-10),(10+int(50*arrow_speed), trajectory_pose_size - 10 - int(10*arrow_turn)),(250,250,250),thickness=2)
        #        
                # Display mean pose
                title = 'trajectory ' + str(n+1)
                trajectory[:,trajectory_pose_size*(t):trajectory_pose_size*(t+1),:] = reconstruction_image
                cv2.imshow(title,trajectory)
                rotator.write(trajectory)
                if cv2.waitKey(int(1)) & 0xFF == ord('q'):
                    break
    rotator.release()


#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                  Plot PCs, Clusters, and Video over time                    -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------
# Set up normalized and data videos, and component data
# -----------------------------------------------------
num_frames = int(mouse_vid.get(cv2.CAP_PROP_FRAME_COUNT))
mouse_vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
#mouse_vid.set(cv2.CAP_PROP_POS_FRAMES,frames[start_frame])
#depth_vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)

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
trajectory_pose_size = 400
show_unchosen_cluster = .15 #threshold to show 2nd-choice cluster
num_PCs_shown = 3

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
if show_PC_plot:
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
        ax_2D.plot(data_for_model_normalized[:,-1-add_turn], color = 'k',linewidth=2) #plot speed
        if add_turn: #plot turn speed or, if available, pose change
            ax_2D.plot(data_for_model_normalized[:,-1], color = 'gray', linestyle = '--',linewidth=2)
    
    # Create legend
    legend_entries = create_legend(num_PCs_shown, add_velocity, True, False, add_turn)
    
    # Create line indicating which frame we're looking at
    center_line = ax_2D.plot([0,0],[-2,2],color = 'gray', linestyle = '--')



# ------------------------------------------------------------
# Show behaviour and 3D videos, with selected pose also shown 
# ------------------------------------------------------------

#j = min(find(frames>=start_frame - window_size*windows_to_look_at + 1)) 
#i = int(frames[j])
#i = start_frame - window_size*windows_to_look_at + 1 #is this right...
i = start_frame - window_size*windows_to_look_at 
j = min(find(frames>=(start_frame - window_size*windows_to_look_at)))
while True:
#    if out_of_bounds[i] == 0:
#        ret1, frame1 = depth_vid.read()         
#    else:
#        chosen_color = [0,0,0]
        
    ret2, frame2 = mouse_vid.read()
    
    if ret2:
        # Grab frame number, which cluster is active, and its corresponding color
        frame_num_mouse = int(mouse_vid.get(cv2.CAP_PROP_POS_FRAMES))
#        frame_num_depth = int(depth_vid.get(cv2.CAP_PROP_POS_FRAMES))
        
        mouse_frame = frame2[:,:,0]
        
        # Display data movie
        if out_of_bounds[i] == 0: #mouse in arena
            chosen_component = chosen_components_to_display[j]
            print('head direction: ' + str(head_dir[i]))
            print('vel forward: ' + str(vel_forward[i]))
            print('vel ortho: ' + str(vel_ortho[i]))
            print('position: ' + str((pos_x[i], pos_y[i])))
            if disruptions[i]:
                print('disrupted')
                print(i)
            print('')
            if only_see and chosen_component != only_see_component:
                j+=1
                i = frame_num_mouse - window_size*windows_to_look_at + 1
                continue
            chosen_color = colors[chosen_component]       
            
            # Grab the images to be displayed
#            depth_frame = frame1[:,:,0]
            
            
            # Resize and recolor images
#            depth_frame = cv2.resize(depth_frame,(450,450))
#            depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
#            depth_frame = (depth_frame * np.squeeze(color_array[:,:,:,chosen_component])).astype(uint8)            

            # Move the PC / clusters plot to be centered at the current frame
            if show_PC_plot:
                center_line.pop(0).remove()
                center_line = ax_2D.plot([j,j],[-2,2],color = 'gray', linestyle = '--')
                ax_2D.set_xlim([j-200,j+200])
                plt.pause(0.01)
                
            #add to trajectory plot
            if show_clusters and not show_clusters_all_at_once:
                ax_3D.scatter(data_for_model_normalized[j,0], data_for_model_normalized[j,1], 
                       data_for_model_normalized[j,-1],color=plot_colors[chosen_component],s=100,alpha=.5)
            j+=1
        else:
            chosen_color = [0,0,0]
#            i = frame_num_mouse - window_size*windows_to_look_at + 1
#            continue

#        elif out_of_bounds[i] == 1: #mouse in shelter
#            depth_frame = shelter
            
        # update current frame index
        #j = frame_num_depth - window_size*windows_to_look_at + 1
        i = frame_num_mouse - window_size*windows_to_look_at + 1  
        if out_of_bounds[i-1] != 0:
            continue
        
        # Add colored circle and frame number to behaviour image
        mouse_frame = cv2.cvtColor(mouse_frame, cv2.COLOR_GRAY2BGR)
        cv2.circle(mouse_frame,(50,50),radius=25, color=chosen_color,thickness=50)
        cv2.putText(mouse_frame,str(i),(50,1000),0,1,255)

        # Display images
        cv2.imshow('behaviour',mouse_frame)
#        cv2.imshow('depth',depth_frame)
        

        
        
        
        # stop video when donw
        if (frame_num_mouse)%500==0:
            print(str(frame_num_mouse) + ' out of ' + str(num_frames) + ' frames complete')   
        if mouse_vid.get(cv2.CAP_PROP_POS_FRAMES) >= min(stop_frame, num_frames):
            break 
        if cv2.waitKey(int(1000/frame_rate)) & 0xFF == ord('q'):
            break
    else:
        print('Problem with Video Playback...')
        cv2.waitKey(int(500))
      
#depth_vid.release()
