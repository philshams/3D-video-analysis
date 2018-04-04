'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
 
-----------#                                        Re-encode a saved video as XVID                            --------------------------------
 
 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
 
import cv2
 
 
#%% ------------------------------------------
# Select video file name and folder location
# ------------------------------------------
old_file_name = 'bj148b_test2.avi' #
new_file_name = 'bj148b_test2_xvid.avi' #
file_loc = 'C:\\Drive\\Video Analysis\\data\\21.03.2018_zina\\'


# Old and New Video Settings 
display_frame_rate = 1000
re_encoded_video_frame_rate = 20
show_old_video = True





 
#%% --------------
# Set up video
# ----------------
vid = cv2.VideoCapture(file_loc + old_file_name) 
end_frame = vid.get(cv2.CAP_PROP_FRAME_COUNT)
print(str(end_frame) + 'frames')

width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

#reencode and save video
file_location_save = file_loc + new_file_name
fourcc = cv2.VideoWriter_fourcc(*'XVID') #LJPG for lossless, XVID or MJPG works for compressed
new_video = cv2.VideoWriter(file_location_save, fourcc , re_encoded_video_frame_rate, (width,height)) 


#%% --------------
# Play/Save video
# ----------------
while True:
    ret, frame = vid.read() # get the frame
 
    if ret:
        new_video.write(frame)
       
        frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)

        if show_old_video:        
            cv2.imshow('movie',frame)
            if cv2.waitKey(int(1000/display_frame_rate)) & 0xFF == ord('q'):
                break
       
        if (frame_num-start_frame)%500==0:
            print(str(int(frame_num-start_frame)) + ' out of ' + str(int(end_frame-start_frame)) + ' frames complete')
           
        if frame_num >= end_frame:
            break
    else:
        print('Problem with movie playback')
#        print(frame_num)
  
new_video.release()
cv2.waitKey()
vid.release()    
cv2.destroyAllWindows()
# Display number of last frame
print('Stopped at frame ' + str(frame_num)) #show the frame number stopped at