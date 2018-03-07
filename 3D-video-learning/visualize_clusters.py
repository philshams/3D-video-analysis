#Visualize Clusters!

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib as mpl
from learning_funcs import reconstruct_from_wavelet
import sys

#Receive data video
file_loc = 'C:\Drive\Video Analysis\data\\'
date = '05.02.2018\\'
mouse_session = '202-1a\\'
file_loc = file_loc + date + mouse_session

save_vid_name = 'analyze_3_5'

depth_vid = cv2.VideoCapture(file_loc + save_vid_name + '_data.avi')   
mouse_vid = cv2.VideoCapture(file_loc + save_vid_name + 'normalized_video.avi')  
frame_rate = 1000
start_frame = 1800
stop_frame = 20000

dims_used = 6

chosen_components = np.load(file_loc+'chosen_components.npy')
plt.close('all')

colors = [[0,0,255],[169,118,14],[10,205,10],[160,0,120],[0,60,100],[0,140,140],[100,100,100],[170,120,220]]
plot_colors = ['red','deepskyblue','green','blueviolet','saddlebrown','yellow','white','lightpink']

#%%

#create coloring arrays:
color_array = np.zeros((450,450,3,len(colors)))
for c in range(len(colors)):
    for i in range(3): #B, G, R
        color_array[:,:,i,c] = np.ones((450,450)) * colors[c][i] / sum(colors[c])
    
    
#Plot mean of clusters
mean_pca_coeffs_model = gmm.means_
var_pca_coeffs_model = gmm.covariances_

#fig_emp = plt.figure('empirical figure',figsize=(30,10))
#ax1 = fig_emp.add_subplot(1,1,1)   
#plt.xlim([1,6])     
#plt.ylim([-1,1])    

fig_mod = plt.figure('model figure',figsize=(30,10))
ax2 = fig_mod.add_subplot(1,1,1)
plt.xlim([1,num_components+1])  
plt.ylim([-1.2,1.2]) 

mean_pca_coeffs_array = np.zeros((normalized_pca_coeffs.shape[1],num_components,2))
var_pca_coeffs_array = np.zeros((normalized_pca_coeffs.shape[1],num_components,2))
#
for n in range(num_components):
    
    for src in [1]:
        if src == 0:
            components_segregated = data_to_fit_gmm[components_over_time[:,n]!=0,:]
            mean_pca_coeffs = np.mean(components_segregated,axis=0)
            mean_pca_coeffs_array[:,n,0] = mean_pca_coeffs
            
            title = 'empirical wavelet reconstruction cluster ' + str(n)
            ax1.add_patch(mpl.patches.Rectangle((n+1, -2), 1,4,color=plot_colors[n],alpha=0.3))
            
        else:
            mean_pca_coeffs = mean_pca_coeffs_model[n,:]
            mean_pca_coeffs_array[:,n,1] = mean_pca_coeffs
            for PC in range(normalized_pca_coeffs.shape[1]):   
                var_pca_coeffs_array[PC,n,1] = var_pca_coeffs_model[n,PC,PC]
            
            title = 'model wavelet reconstruction cluster ' + str(n)
            ax2.add_patch(mpl.patches.Rectangle((n+1, -2), 1,4,color=plot_colors[n],alpha=0.3)) 
            

        mean_wavelet_relevant_features = pca.inverse_transform(np.append(mean_pca_coeffs[0:dims_used],np.zeros(12-dims_used)))
        mean_wavelet = np.zeros(39*39)
        mean_wavelet[relevant_ind] = mean_wavelet_relevant_features
        mean_wavelet_array = np.reshape(mean_wavelet,(39,39))
         
        #reconstruct image from wavelet transform
        reconstruction_from_wavelet  = reconstruct_from_wavelet(mean_wavelet_array,coeff_slices, level, discard_scale)
        reconstruction_image= cv2.resize(abs(reconstruction_from_wavelet).astype(np.uint8),(450,450))
        reconstruction_image= cv2.cvtColor(reconstruction_image, cv2.COLOR_GRAY2BGR)
        reconstruction_image = (reconstruction_image * np.squeeze(color_array[:,:,:,n])).astype(uint8)
        
        arrow_height = mean_pca_coeffs_array[-2,n,src] / max_value[-2]
        arrow_sideness = ((mean_pca_coeffs_array[-1,n,src] / max_value[-1]) + 1) / 2
        cv2.arrowedLine(reconstruction_image,(50,400),(50+int(50*arrow_sideness), 400 - int(50*arrow_height)),(250,250,250),thickness=2)
        
        
        cv2.imshow(title,reconstruction_image)
    
#show PCs
for src in [1]:
    for PC in range(3):
        if src == 0:
            plt.figure('empirical figure')
            
        else:
            plt.figure('model figure')
            
        max_value = np.max(abs(mean_pca_coeffs_array[:,:,src]),axis=1)
        
        plt.scatter(np.arange(num_components)+1.5,mean_pca_coeffs_array[PC,:,src] / max_value[PC],
                    s=100*(normalized_pca_coeffs.shape[1] + 1 - PC), color=PC*np.array([0.2, 0.2, 0.2]))
        
        plt.errorbar(np.arange(num_components)+1.5,mean_pca_coeffs_array[PC,:,src] / max_value[PC], 
                    yerr = var_pca_coeffs_array[PC,:,src] / np.max(abs(mean_pca_coeffs_array[:,:,src]))**2,
                    color=PC*np.array([0.2, 0.2, 0.2]),linewidth = (normalized_pca_coeffs.shape[1] + 1 - PC))
        
    for vel in [-2,-1]:
        if src == 0:
            plt.figure('empirical figure')
            
        else:
            plt.figure('model figure')
            
        max_value = np.max(abs(mean_pca_coeffs_array[:,:,src]),axis=1)
        
        plt.scatter(np.arange(num_components)+1.5,mean_pca_coeffs_array[vel,:,src] / max_value[vel],
                    s=500,color= -vel*np.array([0.2, 0.2, 0.0]))
        
        plt.errorbar(np.arange(num_components)+1.5,mean_pca_coeffs_array[vel,:,src] / max_value[vel], 
                    yerr = var_pca_coeffs_array[PC,:,src] / np.max(abs(mean_pca_coeffs_array[:,:,src]))**2,
                    color=(3+vel)*np.array([0.2, 0.2, 0.0]),linewidth = (-vel*3),linestyle = '--')
              
            
    

#%%

num_frames = int(depth_vid.get(cv2.CAP_PROP_FRAME_COUNT))
depth_vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
mouse_vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)

#plot the PCs
plt.style.use('classic')
fig = plt.figure('PCs',figsize=(30,10))
ax = fig.add_subplot(1,1,1)


for n in range(num_components):
    component_frames = find(components_over_time[:,n])
    plt.scatter(component_frames,np.ones(len(component_frames))*1,color=plot_colors[n],alpha=.7,marker='|',s=700)
    
#    confident_frames = find((chosen_probabilities<.8)*(chosen_components==n))
#    plt.scatter(confident_frames,np.ones(len(confident_frames))*.9,color='k',alpha=.2,marker='|',s=500)
    
    unchosen_component_frames = find(unchosen_components_over_time[:,n] * (unchosen_probabilities>.1))
    plt.scatter(unchosen_component_frames,np.ones(len(unchosen_component_frames))*.9,color=plot_colors[n],alpha=.4,marker='|',s=700)

plt.plot(normalized_pca_coeffs[:,0:3],linewidth=2)
plt.plot(normalized_pca_coeffs[:,-2] * 2, color = 'k',linewidth=2)
plt.plot(normalized_pca_coeffs[:,-1] * 2, color = 'gray', linestyle = '--',linewidth=2)
legend = plt.legend(('PC1','PC2','PC3','direct velocity','ortho velocity'))
center_line = plt.plot([0,0],[-2,2],color = 'gray', linestyle = '--')
plt.title('Principal Components over Time')
plt.xlabel('frame no.')
plt.ylabel('PC amplitude')


i = start_frame
while True:
    ret, frame1 = depth_vid.read() # get the frame
    ret, frame2 = mouse_vid.read()
    
    if ret: 
        frame_num = int(depth_vid.get(cv2.CAP_PROP_POS_FRAMES))
        depth_frame = frame1[:,:,0]
        mouse_frame = frame2[:,:,0]
        
        chosen_component = chosen_components[i]
        chosen_color = colors[chosen_component]
        
        depth_frame = cv2.resize(depth_frame,(450,450))
        
        depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
        mouse_frame = cv2.cvtColor(mouse_frame, cv2.COLOR_GRAY2BGR)
        
        depth_frame = (depth_frame * np.squeeze(color_array[:,:,:,chosen_component])).astype(uint8)
        #cv2.circle(depth_frame,(50,50),radius=25, color=chosen_color,thickness=50)
        cv2.circle(mouse_frame,(50,50),radius=25, color=chosen_color,thickness=50)
        
        cv2.putText(mouse_frame,str(i),(50,400),0,1,255)
        cv2.imshow('depth',depth_frame)
        cv2.imshow('behaviour',mouse_frame)
        
        
        #plot PCs and clusters over time
        center_line.pop(0).remove()
        center_line = plt.plot([i,i],[-2,2],color = 'gray', linestyle = '--')
        plt.xlim([i-750,i+750])
        plt.ylim([-1,1.05])
        plt.pause(0.001)
        

        
        i+=1
        
        if (frame_num)%500==0:
            print(str(frame_num) + ' out of ' + str(num_frames) + ' frames complete')   
        if depth_vid.get(cv2.CAP_PROP_POS_FRAMES) >= stop_frame or depth_vid.get(cv2.CAP_PROP_POS_FRAMES) >= num_frames:
            break 
        if cv2.waitKey(int(1000/frame_rate)) & 0xFF == ord('q'):
            break

        
    else:
        print('broken...')
        
depth_vid.release()