'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                                Turn Data Upside-Down                             --------------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np; import pywt; import cv2; import os


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
file_location = 'C:\Drive\Video Analysis\data\\'
data_folder = 'baseline_analysis\\'
analysis_folder = 'together_for_model\\'
session_name_tags = ['normal_1_0','normal_1_1','normal_1_2',
                    'normal_2_0','normal_2_1','normal_2_2',
                    'normal_3_0','normal_3_1','normal_3_2',
                    'normal_4_0','normal_4_1','normal_4_2',
                    'normal_5_0','normal_5_1',
                    'normal_6_0','normal_6_1','normal_6_2',
                    'clicks_1_0','clicks_2_0','clicks_3_0',
                    'post_clicks_1_0','post_clicks_2_0','post_clicks_3_0']

file_location = file_location + data_folder + analysis_folder

# ---------------------------
# Select analysis parameters
# ---------------------------
display_frame_rate = 1000
frame_rate = 40
end_frame = np.inf
show_images = False
save_data = True
do_not_overwrite = False




#%% -------------------------------------------------------------------------------------------------------------------------------------
#-----------------------                            Turn Upside-Down                       --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# ----------------------
# Set up video playback
# ----------------------
for v in range(len(session_name_tags)):
    session_name_tag = session_name_tags[v]
    file_location_vid = file_location + session_name_tag + '\\' + session_name_tag
    print(file_location_vid)
    vid = cv2.VideoCapture(file_location_vid + '_data.avi')
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    stop_frame = int(np.min([end_frame, num_frames]))
    
    if save_data:
        fourcc = cv2.VideoWriter_fourcc(*'LJPG') #LJPG for lossless
        data_file = file_location_vid + '_data_upside_down.avi'
        if os.path.isfile(data_file) and do_not_overwrite:
            raise Exception('File already exists') 
        upside_down_video = cv2.VideoWriter(data_file,fourcc , frame_rate, (150,150), False) 

       
    
    # ---------------------------------------------
    # for each frame, perform wavelet decomposition
    # ---------------------------------------------
    while True:
        ret, frame = vid.read() # get the frame
    
        if ret: 
            # ---------------------------------------------
            # grab and, optionally, display each frame
            # ---------------------------------------------
            frame_num = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
            frame = frame[:,:,0]
            
            if width != 150:
                frame = cv2.resize(frame,(150,150)) #resize
            if show_images:
                cv2.imshow('normal image',frame)
            
            # -------------
            # rotate image
            # -------------
            M = cv2.getRotationMatrix2D((int(150/2),int(150/2)),180,1) 
            frame = cv2.warpAffine(frame,M,(150,150)) 
            
            if show_images:
                cv2.imshow('the upside-down',frame)
            
            if save_data:
                upside_down_video.write(frame)

            
            # ----------------------------------------------------
            # Stop video when finished and notify every 500 frames
            # ----------------------------------------------------
            if (frame_num)%10000==0:
                print(str(frame_num) + ' out of ' + str(num_frames) + ' frames complete')  
                
            if show_images:
                if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
                    break
                
            if vid.get(cv2.CAP_PROP_POS_FRAMES) >= stop_frame:
                break 
            
        else:
            print('broken...')
            
    # ----------
    # Save data
    # ----------
    vid.release()
    upside_down_video.release()


