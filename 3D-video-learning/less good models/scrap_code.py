'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------#                                          Scrap Code                             --------------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''



#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                       Pre-process scrap                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------
#         except:
#            out_of_bounds_cur = np.load(file_location_saved_data + '_out_of_bounds.npy')
#            out_of_bounds_new = np.zeros((len(out_of_bounds_cur),2))
#            out_of_bounds_new[:,0] = out_of_bounds_cur
#            np.save(file_location_saved_data + '_out_of_bounds.npy', out_of_bounds_new)
#time audit

#t0 = time.time() #get time
#        t_frame = time.time()
#        print('')
#            t_normalize = time.time()
#            print('time to normalize: ' + str(int(1000*(t_normalize-t_frame))))
#            t_undistort = time.time()
#            print('time to undistort norm: ' + str(int(1000*(t_undistort-t_normalize))))
#            t_dilate = time.time()
#            print('time to dilate: ' + str(int(1000*(t_dilate-t_undistort))))
#            t_contour = time.time()
#            print('time to extract contours and check for obscurity: ' + str(int(1000*(t_contour - t_dilate))))
#            t_mask1 = time.time()
#            print('time to make mask: ' + str(int(1000*(t_mask1- t_contour))))
#            t_mask2 = time.time()
#            print('time to apply mask: ' + str(int(1000*(t_mask2- t_mask1))))
#            t_mask3 = time.time()
#            print('time to crop: ' + str(int(1000*(t_mask3- t_mask2))))

       # ----------------------
        # Get mouse orientation from head being higher
        # ----------------------
        
#            # flip mouse into the correct orientation
#            rotate_angle, face_left, ellipse, topright_or_botleft, ellipse_width = \
#            flip_mouse(face_left, ellipse, topright_or_botleft, stereo_image_smoothed, sausage_thresh = 1.1)
#            M = cv2.getRotationMatrix2D((int(crop_size/2),int(crop_size/2)),rotate_angle,1) 
#            stereo_image_straight = cv2.warpAffine(stereo_image_smoothed,M,(crop_size,crop_size))        
#    
#            # check for errors -- if the tail end is more elevated, or the mouse is running toward its tail, slip 180 degrees
#            stereo_top = stereo_image_straight[0:int(crop_size/2),:]
#            stereo_bottom = stereo_image_straight[int(crop_size/2):]
#            rotate_angle, face_left, depth_ratio, history_x, history_y, x_tip, y_tip, flip = \
#            correct_flip(frame_num - start_frame, face_left, stereo_top,stereo_bottom, depth_percentile, depth_ratio, history_x, history_y, cxL, cyL, ellipse, ellipse_width, \
#                     width_thresh=width_thresh, speed_thresh=speed_thresh, depth_ratio_thresh = depth_ratio_thresh, pixel_value_thresh = pixel_value_thresh)   
#            if flip:
#                print('frame ' + str(frame_num-start_frame))
#            M = cv2.getRotationMatrix2D((int(crop_size/2),int(crop_size/2)),rotate_angle,1)
#            stereo_image_straight = cv2.warpAffine(stereo_image_smoothed,M,square_size)
#                            
#            if save_2D_data: # get cropped mouse straight as well
#                frame_L_masked_cropped = cv2.warpAffine(frame_L_masked_cropped,M,square_size)
#                frame_R_masked_cropped = cv2.warpAffine(frame_R_masked_cropped,M,square_size)
        
        
#            else: #head is higher in 3D videos
#        major_top_avg = np.percentile(image_top[image_top>0], depth_percentile) 
#        major_bottom_avg = np.percentile(image_bottom[image_bottom>0], depth_percentile) 
#        depth_ratio[0:2] = depth_ratio[1:3]
#        depth_ratio[2] = major_top_avg / major_bottom_avg
#        
#        if ellipse_width > width_thresh and ( (np.median(depth_ratio) < depth_ratio_thresh[0] and major_bottom_avg > pixel_value_thresh[0]) \
#                                                   or (np.median(depth_ratio) < depth_ratio_thresh[1] and major_bottom_avg > pixel_value_thresh[1]) \
#                                                   or (np.median(depth_ratio) < depth_ratio_thresh[2] and major_bottom_avg > pixel_value_thresh[2])):
#            face_left *= -1
#            flip = 1
#            print('face_HEIGHT_correction!')
#            print(depth_ratio)
#            if major_bottom_avg < pixel_value_thresh[0]:
#                print('at low pixel value' + str(major_bottom_avg))
#            print('')
#    
#local matcher code:
#stereo_left = cv2.StereoBM_create()
#stereo_left.setMinDisparity(64)
#stereo_left.setNumDisparities(num_disparities)
#stereo_left.setBlockSize(SADws)
##stereo_left.setDisp12MaxDiff(50)  #50
##stereo_left.setUniquenessRatio(1)  #10
##stereo_left.setPreFilterSize(5)
##stereo_left.setPreFilterCap(25)
#stereo_left.setTextureThreshold(500)
##stereo_left.setSpeckleWindowSize(200) #or off?
##stereo_left.setSpeckleRange(31)  #or off?
#stereo_left.setPreFilterType(0) #STEREO_BM_XSOBEL; 1 may be   STEREO BM NORMALIZED RESPONSE)
    
#weight least squares filter code:
#wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_left)
#wls_filter.setLambda(lmbda)
#wls_filter.setSigmaColor(sigma)   
#wls_filter.setDepthDiscontinuityRadius(disc_rad)
    
#ROI code:
#    select_roi = False
#if select_roi:
#    ret, frame = vid.read()
#    roi = cv2.selectROI(frame)
#    roi = np.array(roi).astype(int)
#else:
#    try:
#        roi[3]
#        if use_roi == False:
#            roi = [0,0,width,height]
#    except:
#        roi = [0,0,width,height]
#        
#background_mat = background_mat[roi[1]:roi[1]+roi[3], :, :]
    
#put square around roi
        #extract square around mouse
#        blank = np.zeros(frame_norm_L.shape).astype(np.uint8)
#        frame_norm_L_mask_square = cv2.rectangle(blank,(cxL-square_size,cyL-square_size),(cxL+square_size,cyL+square_size),thickness = -1,color = 1)
#        blank = np.zeros(frame_norm_L.shape).astype(np.uint8)
#        frame_norm_R_mask_square = cv2.rectangle(blank,(cxR-square_size,cyR-square_size),(cxR+square_size,cyR+square_size),thickness = -1,color = 1)
#        
#        frame_norm_L_masked = frame_norm_L * frame_norm_L_mask_square
#        frame_norm_R_masked = frame_norm_R * frame_norm_R_mask_square
#        
#        frame_norm_L_masked2 = frame_L * frame_norm_L_mask_square
#        frame_norm_R_masked2 = frame_R_shift * frame_norm_R_mask_square
        
#        stereo_image_L = stereo_left.compute(frame_norm_L_masked2,frame_norm_R_masked2).astype(np.uint8)
#        stereo_image_R = stereo_right.compute(frame_norm_R_masked2,frame_norm_L_masked2).astype(np.uint8)

#code for circle drawing
#cv2.circle(frame_norm_R_masked, (cxR, cyR), radius=3, color=(255,255,255),thickness=5) 
#cv2.circle(stereo_image_filtered_masked, (int((cxR+cxL)/2), int((cyR+cyL)/2)), radius=3, color=(255,255,255),thickness=3) 
  
#code for angle analysis
#print('L angle = ' + str(np.arctan(vyL/vxL)) + 'R angle = ' + str(np.arctan(vyR/vxR)))
#if (np.arctan(vyL/vxL) - np.arctan(vyR/vxR)) < 1.3: #less than 75 deg off
        
#code for ellipse over mouse
        #        #get spine slope of mouse
#        _, contours_stereo, _ = cv2.findContours(stereo_image_cropped_combined_gauss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#       
#        #calculate slope to get spine alignment of mouse
#        corr = np.corrcoef(np.squeeze(contours_stereo[0]).T)        
#        x = 225
#        y = 225
#        if np.abs(corr[0,1]) > .031:  #if corr coeff under certain value, use previous slope
#            [vxR,vyR,x,y] = cv2.fitLine(cntR, cv2.DIST_L2,0,0.01,0.01)
#            [vxL,vyL,x,y] = cv2.fitLine(cntL, cv2.DIST_L2,0,0.01,0.01)
#            slope_recipr = np.mean([(vxL/vyL),(vxR/vyR)])
        #stereo_image_cropped_combined_gauss = cv2.line(stereo_image_cropped_combined_gauss,(int(225-225*(slope_recipr)),0),(int(225+225*(slope_recipr)),450),(100,10,10),3)
        
        
        #head-dir debug prints
#        print('head-up stats')
#        print(depth_ratio)
#        print(major_bottom_avg)
#        print(major_top_avg)
#        print(ellipse_width)       
#        print('x,y movement')
#        print(delta_x)
#        print(delta_y)
#        print('speed')
#        print(speed)
#        print('heading')
#        print(heading)
#        
#        print('put head dir')
#        print(head_dir_putative)
#        
#        print('dot_product')
#        print(np.dot(heading,head_dir_putative))
#        print('')
        
        
#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                       Learning scrap                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

    

''' -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------              Plot PCs and Clusters over Time                            --------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------'''

#if plot_result:
#
#
#    plt.close('all')
#    # reload relevant data
#    components_binary = np.load(file_location_data_library + '_components_binary_' + model_type_and_name_tag + '.npy')
#    unchosen_components_binary = np.load(
#        file_location_data_library + '_unchosen_components_binary_' + model_type_and_name_tag + '.npy')
#    unchosen_probabilities = np.load(file_location_data_library + '_unchosen_probabilities_' + model_type_and_name_tag + '.npy')
#
#    # set up plot
#    plot_colors = ['red', 'deepskyblue', 'green', 'blueviolet', 'saddlebrown', 'lightpink', 'yellow', 'white']
#    set_up_PC_cluster_plot(figsize=(30, 10), xlim=[0, 1000])
#
#    # plot PCs
#    data_for_model_normalized = data_for_model / np.max(data_for_model)  # set max to 1 for visualization purposes
#    np.save(file_location_data_library + '_data_for_' + model_type + '_normalized', data_for_model_normalized)
#    plt.plot(data_for_model_normalized[:, 0:num_PCs_shown])
#
#    # plot velocity and/or pose change
#    if use_speed_as_pseudo_PC:
#        plt.plot(data_for_model_normalized[:, -1 - use_angular_speed_as_pseudo_PC], color='k', linewidth=2)  # plot speed
#        if use_angular_speed_as_pseudo_PC:  # plot turn speed
#            plt.plot(data_for_model_normalized[:, -1], color='gray', linestyle='--', linewidth=2)
#
#    # plot raster of clusters above PCs
#    for n in range(num_clusters):
#        # plot chosen pose:
#        component_frames = np.where(components_binary[:, n])[0]
#        plt.scatter(component_frames, np.ones(len(component_frames)) * 1, color=plot_colors[n], alpha=.7, marker='|',
#                    s=700)
#        # plot 2nd-place pose:
#        unchosen_component_frames = np.where(unchosen_components_binary[:, n] * \
#                                             (unchosen_probabilities > show_unchosen_cluster))[0]
#        plt.scatter(unchosen_component_frames, np.ones(len(unchosen_component_frames)) * .95, color=plot_colors[n],
#                    alpha=.4, marker='|', s=700)
#
#    # Create legend
#    legend_entries = create_legend(num_PCs_shown, use_speed_as_pseudo_PC, True, False, use_angular_speed_as_pseudo_PC)

    
    #super-clustter pie chart
    
#            
## -------------------------------------
## Display Pie Chart for each behaviour
## -------------------------------------
#plt.close('all')
#mean_responsibilities_model = cluster_model_seq.means_
#
#pose_labels = np.array([])
#for p in range(probabilities.shape[1]):
#    pose_labels = np.append(pose_labels, ['pose ' + str(p+1)])
#    
#index_addition1 = probabilities.shape[1]*sequence_based #just look at poses, not past and future onesindex_addition = probabilities.shape[1]*sequence_based*windows_to_look_at
#index_addition2 = probabilities.shape[1]*sequence_based*windows_to_look_at
#for n in range(mean_responsibilities_model.shape[0]):
#    fig = plt.figure('sequence' + str(n+1),figsize=(5,5))
#    plt.pie(mean_responsibilities_model[n,index_addition2:index_addition1+index_addition2] / 
#            np.sum(mean_responsibilities_model[n,index_addition2:index_addition1+index_addition2]), 
#            labels=pose_labels, colors=plot_colors[0:probabilities.shape[1]],shadow=True)
#     
    
    
#    #%% -------------------------------------------------------------------------------------------------------------------------------------
##------------------------              Get distribution over sub-clusters                         --------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------
#
#if display_sub_clusters:
#    # calculate secondary clusters based on threshold set above
#    dual_components = (chosen_components+1)*10+(unchosen_components+1)*(unchosen_probabilities>show_unchosen_cluster)
#    
#    # Create labels for histogram
#    labels = np.array([])
#    for n in range(num_clusters):
#        labels = np.append(labels,(10*(n+1) + arange(num_clusters+1)))
#    
#    # Plot data
#    plt.figure(figsize=(40,7))
#    for n in range(num_clusters):
#        plt.hist(dual_components[(dual_components>=(n+1)*10) * (dual_components<(n+2)*10)],label=labels.astype(str),bins = np.arange(10,61),density = True, color = plot_colors[n])
#    
#    # Format plot
#    plt.xlim([10,(num_clusters+1)*10])
#    plt.ylim([0,.8])
#    plt.xticks(labels)
#    plt.title('Distribution of clusters and sub-clusters')
#    


    # Transition code:
    
    #%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Get Transition Probabilities                         --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------
#
## Get counts of (cluster_preceding --> cluster_following) to (i,j) of cluster_counts
#cluster_counts = np.zeros((num_clusters,num_clusters), dtype=int)
#np.add.at(cluster_counts, (chosen_components[:-1], chosen_components[1:]), 1)
#
## Get transition probs (% in column 1 going to row 1,2,3,etc.)
#transition_prob = (cluster_counts.T / np.sum(cluster_counts,axis=1)).T 
#np.nan_to_num(transition_prob,copy=False)
#
## print transition probs
#print((1000*transition_prob).astype(int)/10)
#
#
## Update probabilities based on transitions:
#updated_probabilities = np.zeros(probabilities.shape)
#probabilities_for_update = probabilities
#components_over_time_for_update = components_over_time
#
#for i in range(3):
#    
#    for n in range(probabilities.shape[1]):
#        
#        updated_probabilities[np.append(False,components_over_time_for_update[:-1,n]).astype(bool),:] \
#        = probabilities_for_update[np.append(False,components_over_time_for_update[:-1,n].astype(bool)),:] * transition_prob[:,n]**0
#        
#        updated_probabilities[0,:] = probabilities[0,:]
#        updated_chosen_components = np.argmax(updated_probabilities,axis=1)
#        
#        
#    #get binarized version of chosen_components and unchosen_components, for later analysis
#    updated_components_over_time = np.zeros((data_to_fit_gmm.shape[0],num_clusters))
#    unchosen_components_over_time = np.zeros((data_to_fit_gmm.shape[0],num_clusters))
#    for n in range(num_clusters):
#        updated_components_over_time[:,n] = (updated_chosen_components == n)
#        unchosen_components_over_time[:,n] = (unchosen_components == n)
#        
#    probabilities_for_update = updated_probabilities    
#    components_over_time_for_update = updated_components_over_time
#
#
#
## Get counts of (cluster_preceding --> cluster_following) to (i,j) of cluster_counts
#cluster_counts = np.zeros((num_clusters,num_clusters), dtype=int)
#np.add.at(cluster_counts, (updated_chosen_components[:-1], updated_chosen_components[1:]), 1)
#
## Get transition probs (% in column 1 going to row 1,2,3,etc.)
#transition_prob = (cluster_counts.T / np.sum(cluster_counts,axis=1)).T 
#np.nan_to_num(transition_prob,copy=False)
#
## print transition probs
#print((1000*transition_prob).astype(int)/10)
#print(sum(components_over_time!=updated_components_over_time))
#
#
#
## set plot parameters
#plot_colors = ['red','deepskyblue','green','blueviolet','saddlebrown','lightpink','yellow','white']
#plt.style.use('classic')
#plt.figure(figsize=(30,10))
#
## plot PCs
#plt.plot(normalized_pca_coeffs[:,0:3])
#if add_velocity:
#    plt.plot(normalized_pca_coeffs[:,-2] * 2, color = 'k',linewidth=2)
#    plt.plot(normalized_pca_coeffs[:,-1] * 2, color = 'gray', linestyle = '--',linewidth=2)
#    
## plot raster of clusters above PCs
#for n in range(num_clusters):
#    component_frames = find(updated_components_over_time[:,n])
#    plt.scatter(component_frames,np.ones(len(component_frames))*1,color=plot_colors[n],alpha=.7,marker='|',s=700)
#    
#    unchosen_component_frames = find(unchosen_components_over_time[:,n] * (unchosen_probabilities>show_unchosen_cluster))
#    plt.scatter(unchosen_component_frames,np.ones(len(unchosen_component_frames))*.9,color=plot_colors[n],alpha=1,marker='|',s=700)
#
## Format plot
#legend = plt.legend(('PC1','PC2','PC3','direct velocity','ortho velocity'))
#legend.draggable()
#plt.title('Principal Components over Time')
#plt.xlabel('frame no.')
#plt.ylabel('PC amplitude')
#plt.xlim([1500,3500])
#plt.ylim([-1,1.05])


#original seq gmm
    # ------------------------------------------------
# Create array of moving-window-chunked sequences
## ------------------------------------------------
#box_filter = np.ones(window_size) / window_size #create boxcar filter
#data_to_fit_gmm_sequence = np.zeros(probabilities.shape)
#probabilities_for_gmm = probabilities
#probabilities_for_gmm[probabilities<0.01] = 0#for efficiency, set small values to 0
#for cluster in range(data_to_fit_gmm_sequence.shape[1]): # apply boxcar filter
#    data_to_fit_gmm_sequence[:,cluster] = np.convolve(probabilities_for_gmm[:,cluster],box_filter,mode='same'
    

#scrap code
    
    
    #empirical pose mean
    
    
#            if src == 0:
#            components_segregated = data_to_fit_gmm[components_over_time[:,n]!=0,:]
#            mean_pca_coeffs = np.mean(components_segregated,axis=0)
#            mean_pca_coeffs_array[:,n,0] = mean_pca_coeffs
#            
#            title = 'empirical wavelet reconstruction cluster ' + str(n)
#            ax1.add_patch(mpl.patches.Rectangle((n+1, -2), 1,4,color=plot_colors[n],alpha=0.3))
#            
#            
            
            
            
    #dual cluster transition probability
    
#    if dual_transitions:
#    ri = dual_components
#    lookup = np.empty((65+1,), dtype=int)
#    lookup[[10,12,13,14,15,20,21,23,24,25,30,31,32,34,35,40,41,42,43,45,50,51,52,53,54]] = np.arange(25)
#    # translate c, r, s to 0, 1, 2
#    rc = lookup[ri]
#    
    #exponential causal filter code:
    
#        exp_filter = np.append(np.exp(np.arange(-filter_length,1)/tau),np.zeros(filter_length))
#    exp_filter = np.flip(exp_filter / sum(exp_filter),axis=0)
#    
#    
#def pca_project(Y,X_mean,components):
#    # Compute the projection of the input data on the selected components
#    Y_shift = Y.T - X_mean
#    
#    # Compute the expansion coefficients of the data
#    coeffs = components@Y_shift.T
#    reconstruction = components.T@coeffs
#    
#    return reconstruction.astype(float).T + X_mean
#
#
#
#
##test on mouse images
#mouse_array = np.load(file_loc + 'analyzemouse_array.npy').astype(np.uint8)
#mouse_array_linear = np.reshape(mouse_array,(20*20,50))
#mouse_mean, mouse_components = pca_manual(mouse_array_linear,n_components = 10) #input 400 dim x 50 sample array
#
#pca = sklearn.decomposition.PCA(n_components=None, svd_solver = 'full')
#pca.fit(mouse_array_linear.T) #input (samples, features)
#
##pca.components_  #components (component#, features)
##pca.explained_variance_ratio_
#
##mouse_array_recon = pca_project(mouse_array_linear,mouse_mean,mouse_components).astype(np.uint8)
#mouse_array_recon = pca_project(mouse_array_linear,mouse_mean,pca.components_).astype(np.uint8)
#mouse_array_recon = np.reshape(mouse_array_recon.T,(20,20,50))
#
#
#
#cv2.imshow('original',cv2.resize(mouse_array[:,:,10],(300,300)))
#cv2.imshow('reconstruction',cv2.resize(mouse_array_recon[:,:,10],(300,300)))
    

##pca code
#    coeffs = pca.components_[0:n_com+1,:]@(wavelet_array_relevant_features - wavelet_relevant_mean).T
#    wavelet_array_relevant_features_recon = (pca.components_[0:n_com+1].T@coeffs).astype(float).T + wavelet_relevant_mean