'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------#                                          Display a saved video                            --------------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import cv2


# ------------------------------------------
# Select video file name and folder location
# ------------------------------------------
file_name = 'neuropixel_1_data.avi' 
file_loc = 'C:\\Drive\\Video Analysis\\data\\neuropixel\\neuropixel_1\\'
data_file_location = file_loc + 'neuropixel_1_data2.avi'

show_video = True

display_frame_rate = 1000
start_frame = 0
end_frame = np.inf




fourcc_data = cv2.VideoWriter_fourcc(*"LJPG")  # LJPG for lossless, XVID for compressed
data_video = cv2.VideoWriter(data_file_location, fourcc_data, 40, (150, 150), False)



# -----------
# Play video
# -----------
vid = cv2.VideoCapture(file_loc + file_name)  
vid.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)
#end_frame = min(end_frame,vid.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret, frame = vid.read() # get the frame

    if ret:     
        data_video.write(frame[:,:,0])
        frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)

        if show_video:
            cv2.imshow('movie',frame)
            if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
                break
        
            if (frame_num-start_frame)%1000==0:
                print(str(int(frame_num-start_frame)) + ' out of ' + str(int(end_frame-start_frame)) + ' frames complete')
            
        if frame_num >= end_frame:
            break 
    else:
        print('Problem with movie playback')
        cv2.waitKey(1000)
        break
        
vid.release()
# Display number of last frame
print('Stopped at frame ' + str(frame_num)) #show the frame number stopped at
data_video.release()

 