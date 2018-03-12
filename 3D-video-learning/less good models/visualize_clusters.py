'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                           Visualize the Poses Determined by the cluster_model                            -------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib as mpl
from learning_funcs import reconstruct_from_wavelet
from sklearn.externals import joblib
import sys


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


# --------------------------------
# Select visualization parameters
# --------------------------------
frame_rate = 10
start_frame = 1600
stop_frame = 4800

model= 'hmm'

num_PCs_shown = 3
show_unchosen_cluster = .15



#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Load data, models, and video                    -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

#load videos
depth_vid = cv2.VideoCapture(file_loc + '_data.avi')   
mouse_vid = cv2.VideoCapture(file_loc + '_normalized_video.avi')  

#load data
chosen_components = np.load(file_loc+'_chosen_components.npy')
data_to_fit_model_normalized = np.load(file_loc+'_data_for_' + model + '_normalized.npy')
add_velocity, speed_only, add_change, num_PCs_used = np.load(file_loc + '_' + model + '_settings.npy')
relevant_ind = np.load(file_loc + '_wavelet_relevant_ind.npy')
coeff_slices = np.load(file_loc + '_wavelet_slices.npy')
probabilities = np.load(file_loc+'_probabilities.npy')
num_clusters = probabilities.shape[1]

# these must be parameters taken from original wavelet transform 
level = 5
discard_scale = 4

#load models
pca = joblib.load(file_loc + '_pca')
cluster_model = joblib.load(file_loc + '_' + model)





#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Plot Mean Poses                                  -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------
# Get mean and covariance of clusters
# ------------------------------------
mean_pca_coeffs_model = cluster_model.means_
if model == 'hmm':
    var_pca_coeffs_model = cluster_model.covars_
elif model == 'gmm':
    var_pca_coeffs_model = cluster_model.covariances_

mean_pca_coeffs_array = np.zeros((data_to_fit_model_normalized.shape[1],num_clusters))
var_pca_coeffs_array = np.zeros((data_to_fit_model_normalized.shape[1],num_clusters))
   

# -------------
# Set up plots
# -------------

plt.close('all')

fig_mod = plt.figure('model poses',figsize=(30,10))
ax2 = fig_mod.add_subplot(1,1,1)
plt.xlim([1,num_clusters+1])  
plt.ylim([-1.2,1.2]) 

# set up colors
colors = [[0,0,255],[169,118,14],[10,205,10],[160,0,120],[0,80,120],[170,120,220],[0,140,140],[100,100,100]]
plot_colors = ['red','deepskyblue','green','blueviolet','orange','lightpink','yellow','white']

#create coloring arrays:
color_array = np.zeros((450,450,3,len(colors)))
for c in range(len(colors)):
    for i in range(3): #B, G, R
        color_array[:,:,i,c] = np.ones((450,450)) * colors[c][i] / sum(colors[c])
        


# ------------------------------------
# Display mean poses for each cluster
# ------------------------------------
        
for n in range(num_clusters):
        mean_pca_coeffs = mean_pca_coeffs_model[n,:] # Get the mean PC coeffs for that pose
        mean_pca_coeffs_array[:,n] = mean_pca_coeffs # And save to array for the next section
        
        for PC in range(data_to_fit_model_normalized.shape[1]): # Save the variances of each PC as well
            var_pca_coeffs_array[PC,n] = var_pca_coeffs_model[n,PC,PC]
        
        # Add transparent rectangle to plot of PCs
        ax2.add_patch(mpl.patches.Rectangle((n+1, -2), 1,4,color=plot_colors[n],alpha=0.3)) 
        
        # Reconstruct wavelet-transformed data from the PCs
        mean_wavelet_relevant_features = pca.inverse_transform(np.append(mean_pca_coeffs[0:num_PCs_used],np.zeros(12-num_PCs_used)))
        mean_wavelet = np.zeros(39*39)
        mean_wavelet[relevant_ind] = mean_wavelet_relevant_features
        mean_wavelet_array = np.reshape(mean_wavelet,(39,39))
         
        # Reconstruct image in pixel space from wavelet-transformed reconstruction
        reconstruction_from_wavelet  = reconstruct_from_wavelet(mean_wavelet_array,coeff_slices, level, discard_scale)
        reconstruction_image= cv2.resize(abs(reconstruction_from_wavelet).astype(np.uint8),(450,450))
        reconstruction_image= cv2.cvtColor(reconstruction_image, cv2.COLOR_GRAY2BGR)
        reconstruction_image = (reconstruction_image * np.squeeze(color_array[:,:,:,n])).astype(uint8)
        
        # Also display an arrow indicating the mean velocity of that cluster
        if add_velocity:
            arrow_height = mean_pca_coeffs_array[-2-add_change+speed_only,n] / np.max(mean_pca_coeffs_model[:,-2-add_change+speed_only])
            if not(speed_only) or add_change:
                arrow_sideness = ((mean_pca_coeffs_array[-1-add_change+speed_only,n] / np.max(mean_pca_coeffs_model[:,-2+add_change])) + 1) / 2
            else:
                arrow_sideness = 0
            cv2.arrowedLine(reconstruction_image,(50,400),(50+int(50*arrow_sideness), 400 - int(50*arrow_height)),(250,250,250),thickness=2)
        
        # Display mean pose
        title = 'cluster ' + str(n+1)
        cv2.imshow(title,reconstruction_image)
    
    
# -----------------------------------------------------
# Display mean PCs and their variance for each cluster
# -----------------------------------------------------
for PC in range(num_PCs_shown):

    #find the max value of each pc, and normalize to it
    max_value = np.max(abs(mean_pca_coeffs_array[:,:]),axis=1)
    
    #plot PC mean vs cluster
    plt.scatter(np.arange(num_clusters)+1.5,mean_pca_coeffs_array[PC,:] / max_value[PC],
                s=100*(data_to_fit_model_normalized.shape[1] + 1 - PC), color=PC*np.array([0.2, 0.2, 0.2]))
    #plot PC std vs cluster
    plt.errorbar(np.arange(num_clusters)+1.5,mean_pca_coeffs_array[PC,:] / max_value[PC], 
                yerr = np.sqrt(var_pca_coeffs_array[PC,:] / np.max(abs(mean_pca_coeffs_array[:,:]))**2),
                color=PC*np.array([0.2, 0.2, 0.2]),linewidth = (data_to_fit_model_normalized.shape[1] + 1 - PC))

if add_velocity:    
    if add_change:
        vel_ind = [-3,-1]
        legend_change_entry = 'pose change'
    else:
        vel_ind = [-2,-1]
        legend_change_entry = 'ortho speed'
    for vel in vel_ind:
    
        #find the max value of each velocity component, and normalize to it
        max_value = np.max(abs(mean_pca_coeffs_array[:,:]),axis=1)
        
        #plot velocity component mean vs cluster
        plt.scatter(np.arange(num_clusters)+1.5,mean_pca_coeffs_array[vel,:] / max_value[vel],
                    s=500,color= -vel*np.array([0.2, 0.2, 0.0]))
        
        #plot velocity component std vs cluster
        plt.errorbar(np.arange(num_clusters)+1.5,mean_pca_coeffs_array[vel,:] / max_value[vel], 
                    yerr = np.sqrt(var_pca_coeffs_array[PC,:] / np.max(abs(mean_pca_coeffs_array[:,:]))**2),
                    color=(3+vel)*np.array([0.2, 0.2, 0.0]),linewidth = (-vel*3),linestyle = '--')
                  
                
    

#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                  Plot PCs, Clusters, and Video over time                    -----------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------
# Set up normalized and data videos
# -----------------------------------
num_frames = int(depth_vid.get(cv2.CAP_PROP_FRAME_COUNT))
depth_vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
mouse_vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)

# -------------------------------
# Set up and plot moving PC plot
# -------------------------------

plt.style.use('classic')
fig = plt.figure('PCs',figsize=(6,2))
ax = fig.add_subplot(1,1,1)
plt.title('Principal Components over Time')
plt.xlabel('frame no.')
plt.ylabel('PC amplitude')

# Plot the chosen and almost-chosen clusters above the PCs
for n in range(num_clusters):
    component_frames = find(components_over_time[:,n])
    plt.scatter(component_frames,np.ones(len(component_frames))*1,color=plot_colors[n],alpha=.7,marker='|',s=700)
    
    unchosen_component_frames = find(unchosen_components_over_time[:,n] * (unchosen_probabilities>show_unchosen_cluster))
    plt.scatter(unchosen_component_frames,np.ones(len(unchosen_component_frames))*.9,color=plot_colors[n],alpha=.4,marker='|',s=700)

# plot the desired number of pc coefficients
plt.plot(data_to_fit_model_normalized[:,0:num_PCs_shown],linewidth=2)

# plot the velocity
if add_velocity:
    plt.plot(data_to_fit_model_normalized[:,-2-add_change+speed_only], color = 'k',linewidth=2) #plot speed
    if not(speed_only) or add_change: #plot turn speed or, if available, pose change
        plt.plot(data_to_fit_model_normalized[:,-1], color = 'gray', linestyle = '--',linewidth=2)

# create legend and plot line indicating which frame we're looking at
legend_entries = []
for pc in range(num_PCs_shown):
    legend_entries.append('PC' + str(pc))
if add_velocity:
    legend_entries.append('speed')
    if not(speed_only):
        legend_entries.append('turn speed')
if add_change:
    legend_entries.append('change in pose')
legend = plt.legend((legend_entries))
center_line = plt.plot([0,0],[-2,2],color = 'gray', linestyle = '--')


# ------------------------------------------------------------
# Show behaviour and 3D videos, with selected pose also shown 
# ------------------------------------------------------------
i = start_frame
while True:
    ret, frame1 = depth_vid.read() 
    ret, frame2 = mouse_vid.read()
    
    if ret:
        # Grab frame number, which cluster is active, and its corresponding color
        frame_num = int(depth_vid.get(cv2.CAP_PROP_POS_FRAMES))
        chosen_component = chosen_components[i]
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
        center_line = plt.plot([i,i],[-2,2],color = 'gray', linestyle = '--')
        plt.xlim([i-500,i+500])
        plt.ylim([-1,1.05])
        plt.pause(0.02)
        
        # update current frame index
        i+=1
        
        # stop video when donw
        if (frame_num)%500==0:
            print(str(frame_num) + ' out of ' + str(num_frames) + ' frames complete')   
        if depth_vid.get(cv2.CAP_PROP_POS_FRAMES) >= stop_frame or depth_vid.get(cv2.CAP_PROP_POS_FRAMES) >= num_frames-1:
            break 
        if cv2.waitKey(int(1000/frame_rate)) & 0xFF == ord('q'):
            break

    else:
        print('broken...')
        cv2.waitKey(int(100))
        
depth_vid.release()