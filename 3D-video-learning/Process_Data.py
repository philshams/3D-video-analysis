'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                  Straighten, Remove Background, and Perform Wavelet Transform on video        ---------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np; import pywt; import cv2; import os; from learning_funcs import get_biggest_contour, filter_features; from sklearn.externals import joblib


#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Select data file and analysis parameters                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
file_location = 'C:\Drive\Video Analysis\data\\'
data_folder = 'baseline_analysis\\'
analysis_folder = 'together_for_model\\'
session_name_tags = ['normal_1_0','normal_1_1','normal_1_2',
                    'normal_2_0','normal_2_1','normal_2_2',
                    'normal_3_0','normal_3_1',
                    'normal_4_0','normal_4_1','normal_4_2',
                    'normal_5_0','normal_5_1',
                    'normal_6_0','normal_6_1','normal_6_2',
                    'clicks_1_0','clicks_2_0','clicks_3_0',
                    'post_clicks_1_0','post_clicks_2_0','post_clicks_3_0']

concatenated_data_name_tag = 'analyze3' 

   



# ---------------------------
# Select analysis parameters
# ---------------------------
show_images = False
frame_rate = 30
stop_frame = np.inf


straighten_upside_down_frames = True #use existing LDA and wavelet transform to straighten upside-down frames
flip_threshold = .85

do_not_overwrite = False


erode = True; normalize = True; reject = True
erode_iterations = 6 #for erosion: how (note this will also realign orientation angle which can be turned off in the code)
mouse_size_pixels = [1200,5000] #for reject: mouse must be within these sizes, in number of pixels
background_luminance_threshold = 150 #for erode: lower threshold means discarding more pixels; if eroded, this is relative to a mean of 127
width_threshold = 1.3 #ellipticality of ellipse needed for orientation angle changes to be registered




#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                    Do processing & wavelet transform                       ------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------
# Set up position_orientation_velocity concatenated data array
# -------------------------------------------------------------
folder_location_concatenated_data = file_location + data_folder + analysis_folder + concatenated_data_name_tag + '\\'
file_location_concatenated_data = folder_location_concatenated_data + concatenated_data_name_tag   

if straighten_upside_down_frames:
    position_orientation_velocity = np.load(file_location_concatenated_data + '_position_orientation_velocity.npy')
    position_orientation_velocity_new = np.array(([],[],[],[],[],[],[],[])).T
    save_concatenate_data = False
    upside_down = False
    reject = False
    realign = False
elif erode:
    realign = True
else:
    realign = False
    
if save_concatenate_data:
   position_orientation_velocity = np.array(([],[],[],[],[],[],[],[])).T 
    

 
level = 5 # how many different spatial scales to use
discard_scale = 4 # 4 discards 4/5; 6 keeps all

# ----------------------------------
# Loop over the videos listed above
# ----------------------------------
       
for session_num in range(len(session_name_tags)): # <---- use enumerate instead of range(len())
    session_name_tag = session_name_tags[session_num]
    file_location_data = file_location + data_folder + analysis_folder + session_name_tag + '\\' + session_name_tag
    print(file_location_data)
       
    flip_ind = []
    
    # --------------------------------------------------------------
    # Fit the LDA to these data, to find upside-down frames to flip
    # --------------------------------------------------------------
    if straighten_upside_down_frames: #use the LDA to flip flipped frames       
        relevant_ind = np.load(file_location_concatenated_data + '_wavelet_relevant_ind_LDA.npy') 
        wavelet_array = np.load(file_location_data + '_wavelet.npy')
        
        wavelet_array = np.reshape(wavelet_array,(39*39,wavelet_array.shape[2]))
        wavelet_array_relevant_features = wavelet_array[relevant_ind,:].T

        #get velocity
        position_orientation_velocity_cur = position_orientation_velocity[position_orientation_velocity[:,7]==session_num,:]
        mouse_in_bounds_index = position_orientation_velocity_cur[:,1]==0
        frames_of_data = position_orientation_velocity_cur[mouse_in_bounds_index,0]
        velocity = position_orientation_velocity_cur[mouse_in_bounds_index,2:3] #velocity toward head direction is index 2
        orientation_angle = position_orientation_velocity_cur[mouse_in_bounds_index,4]
        
        disruptions = position_orientation_velocity_cur[mouse_in_bounds_index,7]
        
        #rescale data
        velocity_scaler = joblib.load(file_location_concatenated_data + '_lda_velocity_scaling')
        scaling_array = np.load(file_location_concatenated_data + '_lda_scaling.npy') #np.array([[mean_vel],[std_vel],up_means,up_stds])
        mean_vel, std_vel, up_means, up_stds = scaling_array[0][0], scaling_array[1][0], scaling_array[2], scaling_array[3]
        
        velocity[disruptions==1,:] = 0 #remove spurious velocities
        less_than_median_vel_ind = np.squeeze(velocity <= np.median(velocity) + std_vel)
        print('only flip frames below ' + str(np.median(velocity) + std_vel) + ' pixels/frame')
        velocity[velocity[:,0]-mean_vel > 6*std_vel,0] = mean_vel + 6*std_vel #saturate velocity above spurious level
        velocity[velocity[:,0]-mean_vel < -6*std_vel,0] = mean_vel - 6*std_vel #saturate velocity below spurious level
        velocity = velocity_scaler.transform(velocity[:,0:1])
        wavelet_array_relevant_features = (wavelet_array_relevant_features - up_means) / up_stds
        
        # append data and velocity
        features = np.concatenate((wavelet_array_relevant_features,velocity),axis=1)
        
        # fit lda to the features to classify as straight or upside-down
        lda = joblib.load(file_location_concatenated_data + '_lda')
        proportion_straight = lda.score(features, np.ones(features.shape[0]))
        print(str(int(100*(1-proportion_straight))) + '% of trials classified as upside-down')
        
        #calculate probabilities/predictions for a particular session
        predicted_prob_up = lda.predict_proba(features)

        predicted_state_flip = (predicted_prob_up[:,0] > flip_threshold) * less_than_median_vel_ind 
        predicted_state_flip_filtered = np.zeros((len(predicted_state_flip),1))
        predicted_state_flip_filtered[:,0] = predicted_state_flip
        predicted_state_flip_filtered = (filter_features(predicted_state_flip_filtered, 3, np.inf)>.5)[:,0]
        
        flip_ind = find((predicted_state_flip+predicted_state_flip_filtered)>0)
        print(str(int(100*len(flip_ind)/len(predicted_state_flip))) + '% of ' + str(len(predicted_state_flip)) + ' trials flipped after adjustment')
        print('')
        #frames to flip
        wavelet_frames_cur = position_orientation_velocity_cur[mouse_in_bounds_index,0]
        data_frames_cur = np.load(file_location_data + '_frames.npy')
        velocity_cur = np.load(file_location_data + '_velocity.npy')
        frames_to_flip = wavelet_frames_cur[flip_ind].astype(int)
        velocity_flipped = np.ones((len(frames_to_flip),2)) * np.nan
        orientation_angles_flipped = np.ones(len(frames_to_flip)) * np.nan
        
    flip_counter = 0
    
    # ---------------------------------------------------
    # open current additional data to concatenate / flip
    # ---------------------------------------------------
    if save_concatenate_data:
        out_of_bounds_cur = np.load(file_location_data + '_out_of_bounds.npy')
        disruptions_cur = np.load(file_location_data + '_disruption.npy').astype(float)
        disruption = 0
        frames_cur = np.load(file_location_data + '_frames.npy')
        frames_index_cur = frames_cur.copy().astype(int)
        first_frame = int(frames_cur[0])
        velocity_cur = np.load(file_location_data + '_velocity.npy')
        coordinates_cur   = np.load(file_location_data + '_coordinates.npy')  
        
        orientation_angles_cur = np.angle(velocity_cur[:,2] + velocity_cur[:,3]*1j,deg=True)

        position_orientation_velocity_cur = np.ones((out_of_bounds_cur.shape[0],8)) * np.nan
        
    # ---------------------------------------    
    # open the and initialize the data video
    # ---------------------------------------
    vid = cv2.VideoCapture(file_location_data + '_data' + '.avi')
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    end_frame = int(np.min([stop_frame, num_frames]))
    if not straighten_upside_down_frames: #initialize wavelet-transformed data array, unless just modifying an existing one
        wavelet_array = np.zeros((39,39,end_frame,upside_down+1)).astype(np.float16)
         
    # ----------------
    # grab each frame
    # ----------------
    while (not straighten_upside_down_frames) or flip_counter < len(flip_ind):
        if straighten_upside_down_frames:
            data_frame = find(data_frames_cur == frames_to_flip[flip_counter]) #data frame is among all frames saved in _data.npy
            wavelet_frame = find(wavelet_frames_cur == frames_to_flip[flip_counter]) #wavelet frame lacks frames rejected in processing
            vid.set(cv2.CAP_PROP_POS_FRAMES, data_frame)
        ret, frame = vid.read() # get the frame
        
        if ret: 
            frame_num = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
            frame_from_data = frame[:,:,0]
            skip_frame = False
            if width != 150:
                frame_from_data = cv2.resize(frame_from_data,(150,150)) #resize
                
            # ------------------------------------------------------------------------------
            # loop over the upright and inverted versions of the data (if upside_down==True)
            # ------------------------------------------------------------------------------
            for image_flip in range(upside_down+1): 
                if image_flip == 1: #create inverted version of data                 
                    M = cv2.getRotationMatrix2D((int(150/2),int(150/2)),180,1) 
                    frame = cv2.warpAffine(frame_from_data,M,(150,150))  
                else:
                    frame = frame_from_data 
                    
                # --------------------------------------------
                # flip frame before processing, if applicable
                # --------------------------------------------
                if straighten_upside_down_frames:
                    M = cv2.getRotationMatrix2D((int(150/2),int(150/2)),180,1) 
                    frame = cv2.warpAffine(frame,M,(150,150)) 
                    orientation_angles_flipped[flip_counter] = np.angle(-1*velocity_cur[data_frame,2] + -1*velocity_cur[data_frame,3]*1j,deg=True)
                    velocity_flipped[flip_counter,:] = -1 * velocity_cur[data_frame,0:2]                       
                    
                # -------------------------------------------------------------
                # normalize (to mean = 127) data image and erode away background
                # -------------------------------------------------------------
                if normalize: #normalize by the mean pixel value
                    zero_mean_array = frame[frame>0] / np.mean(frame[frame>0]) * 127.5
                    zero_mean_array[zero_mean_array > 250] = 250
                    frame[frame>0] = (zero_mean_array + (127 - np.mean(zero_mean_array))).astype(uint8)
                if erode: #erode away background
                    frame = cv2.erode(frame, np.ones((2,2),np.uint8), iterations=erode_iterations) #change to four 
                    frame[frame>background_luminance_threshold] = 0                
                    contours, big_cnt_ind, _, _, _ = get_biggest_contour(frame)
                    # create a new mask, corresponding to only the largest contour
                    blank = np.zeros(frame.shape).astype(np.uint8)
                    contour_mask = cv2.drawContours(blank, contours, big_cnt_ind, color=(1,1,1), thickness=cv2.FILLED)
                    contour_area = np.sum(contour_mask)  
                    frame = frame * contour_mask

                # ----------------------------------------------
                # throw out frames with a non-mouse-sized object
                # ----------------------------------------------  
                if save_concatenate_data:                 
                    if (sum(frame>0) < mouse_size_pixels[0] or sum(frame>0) > mouse_size_pixels[1]) and reject:
                         disruption = 1
                         disruptions_cur[frame_num-1] = np.nan
                         
                         frame_num_whole_data = frames_cur[frame_num-1].astype(int)
                         frames_cur[frame_num-1] = np.nan
                         velocity_cur[frame_num-1,:] = np.ones(4) * np.nan
                         coordinates_cur[frame_num-1,:] = np.ones(2) * np.nan
                         orientation_angles_cur[frame_num-1] = np.nan
                         
                         out_of_bounds_cur[frame_num_whole_data] = 3
                         wavelet_array[:,:,frame_num-1] = np.nan
                         skip_frame = True
                         break
                    elif disruption==1:
                        disruptions_cur[frame_num-1] = 1
                        disuption = 0
                    
                # ----------------------------------------------
                # straighten mouse, update angle, save to array
                # ----------------------------------------------
                if realign:
                    contours, big_cnt_ind, _, _, _ = get_biggest_contour(frame)
                    ellipse = cv2.fitEllipse(contours[big_cnt_ind])
                    ellipse_angle = ellipse[2]-180*(ellipse[2]>90)
    
                    M = cv2.getRotationMatrix2D((int(150/2),int(150/2)),ellipse_angle,1) 
                    frame = cv2.warpAffine(frame,M,(150,150))                      
                    
                    ellipse_width = (ellipse[1][1] / ellipse[1][0])

                    if ellipse_width > width_threshold:
                        orientation_angles_cur[frame_num-1] = orientation_angles_cur[frame_num-1] - ellipse_angle
                    elif isfinite(orientation_angles_cur[frame_num-2]):
                        orientation_angles_cur[frame_num-1] = orientation_angles_cur[frame_num-2]
                        
                # -----------------------------
                # extract wavelet coefficients
                # -----------------------------
                if do_wavelet_analysis:           
                    coeffs_lowpass = [[],[],[],[],[],[]]
                    coeffs = pywt.wavedec2(frame, wavelet='db1',level = level)
                    for i in range(level+1):
                        #discard coefficients at too coarse of a spaital scale, as set by discard_scale
                        if i < discard_scale:
                            coeffs_lowpass[i] = coeffs[i]
                        else:
                            coeffs_lowpass[i] = [None,None,None]
                    #place coefficients in an array, and take coeff_slices index for later reconstruction       
                    coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs[0:discard_scale])
                    if straighten_upside_down_frames:
                        wavelet_array[:,wavelet_frame] = np.reshape(coeff_array,(39*39,1))      
                        flip_counter += 1
                    else:
                        wavelet_array[:,:,frame_num-1,image_flip] = coeff_array                            

                
            if skip_frame:
                continue
            # --------------
            # Display images
            # --------------
            if show_images:
                wavelet_recon = pywt.waverec2(coeffs_lowpass, wavelet='db1').astype(np.uint8)
                
                cv2.imshow('normal image',frame_from_data)
                cv2.imshow('wavelet reconstruction',wavelet_recon)
                if upside_down or straighten_upside_down_frames:
                    cv2.imshow('flipped and processed image',frame)
                
            # ----------------------------------------------------
            # Stop video when finished and notify every 500 frames
            # ----------------------------------------------------
                if cv2.waitKey(int(1000/frame_rate)) & 0xFF == ord('q'):
                    break
            if (frame_num)%5000==0 and not straighten_upside_down_frames:
                print(str(frame_num) + ' out of ' + str(num_frames) + ' frames complete')        
            if vid.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
                break 
            
        else:
            print('broken...')
            
    # -------------
    # Save wavelets
    # -------------
    vid.release()
    if do_wavelet_analysis:
        if straighten_upside_down_frames:
            if os.path.isfile(file_location_data + '_wavelet_corrected.npy') and do_not_overwrite:
                raise Exception('File already exists')
            wavelet_array = np.reshape(wavelet_array,(39,39,wavelet_array.shape[1]))
            np.save(file_location_data + '_wavelet_corrected.npy', wavelet_array) 
            position_orientation_velocity_cur[frames_to_flip,2:4] = velocity_flipped
            position_orientation_velocity_cur[frames_to_flip,4] = orientation_angles_flipped
            position_orientation_velocity_new = np.concatenate((position_orientation_velocity_new, position_orientation_velocity_cur),axis=0)
        else:
            save_file = file_location_data + '_wavelet.npy'
            if os.path.isfile(save_file) and do_not_overwrite:
                raise Exception('File already exists')
            coeff_save_file = file_location_data + '_wavelet_slices'
            np.save(coeff_save_file,coeff_slices)
            non_nan_ind = ~np.isnan(wavelet_array[0,0,:,0])
            wavelet_array_up = np.squeeze(wavelet_array[:,:,non_nan_ind,0])
            print(str(wavelet_array_up.shape[2]) + ' valid frames saved')
            np.save(file_location_data + '_wavelet.npy', wavelet_array_up)         
            if upside_down:
                save_file_down = file_location_data + '_upside_down_wavelet'
                wavelet_array_down = np.squeeze(wavelet_array[:,:,non_nan_ind,1])
                np.save(save_file_down,wavelet_array_down)
       
# ------------------------------------------
# save concatenated / flipped velocity, etc. 
# ------------------------------------------
    if save_concatenate_data:
        mouse_in_bounds_index = frames_index_cur
        position_orientation_velocity_cur[mouse_in_bounds_index,0] = frames_cur
        position_orientation_velocity_cur[:,1] = out_of_bounds_cur
        position_orientation_velocity_cur[mouse_in_bounds_index,2:4] = velocity_cur[:,0:2]
        position_orientation_velocity_cur[mouse_in_bounds_index,4] = orientation_angles_cur
        position_orientation_velocity_cur[mouse_in_bounds_index,5:7] = coordinates_cur
        position_orientation_velocity_cur[:,7] = session_num
        position_orientation_velocity = np.concatenate((position_orientation_velocity, position_orientation_velocity_cur),axis=0)       
        
if save_concatenate_data:
    if os.path.isfile(file_location_concatenated_data + '_position_orientation_velocity.npy') and do_not_overwrite:
        raise Exception('File already exists')        
    if not os.path.isdir(folder_location_concatenated_data):
        os.makedirs(folder_location_concatenated_data)
    np.save(file_location_concatenated_data + '_position_orientation_velocity.npy', position_orientation_velocity)
    np.save(file_location_concatenated_data + '_session_name_tags.npy', session_name_tags)
    np.save(file_location_concatenated_data + '_wavelet_slices',coeff_slices)
elif straighten_upside_down_frames:
    if os.path.isfile(file_location_concatenated_data + '_position_orientation_velocity_corrected.npy') and do_not_overwrite:
        raise Exception('File already exists') 
    np.save(file_location_concatenated_data + '_position_orientation_velocity_corrected.npy', position_orientation_velocity_new)


    
       

    
    

