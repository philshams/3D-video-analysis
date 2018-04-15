'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------#                     Display and Preprocess a saved video                            --------------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np
import cv2
import os
from depth_funcs import get_background_mean, get_offset, get_biggest_contour, get_second_biggest_contour
from depth_funcs import flip_mouse, correct_flip, write_videos

'''-------------------------------------------------------------------------------------------------------------------------------------
# ------------------------               Select video file and analysis parameters                 --------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------'''


# ------------------------------------------
# Select video file name and folder location
# ------------------------------------------
video_file_location = "C:\\Drive\\Video Analysis\\data\\baseline_analysis\\27.02.2018\\205_1a\\"
video_name = "Chronic_Mantis_stim-default-996386-video-"
videos = [0]

save_folder_location = "C:\\Drive\\Video Analysis\\data\\baseline_analysis\\test_pipeline\\"
session_name_tag = "loom_1"  # name-tag to be associated with all saved files
initial_run = False
use_saved_settings = False


# -----------------------
# Set data-saving options
# -----------------------
show_images = True
save_data = False
save_videos = False
do_not_overwrite = True
frame_rate_online_display = 1000  # 1000 makes it as fast as possible











# -------------------------
# Set processing parameters
# -------------------------
# set processing settings
if not use_saved_settings:
    # --------------------
    # Set video parameters
    # --------------------
    frame_rate_saved_video = 40  # aesthetic frame rate of saved videos
    process_entire_video = True
    if not process_entire_video:
        start_frame = 0  # what frame to start at
        stop_frame = np.inf  # can set to np.inf to go to end
    else:
        start_frame = 0
        stop_frame = np.inf

    # ----------------------------
    # Set mouse contour parameters
    # ----------------------------
    mask_thresh = .42  # mouse mask threshold (lower is more stringently darker than background)
    kernel = [4, 3]  # erosion and dilation kernel sizes for mouse mask
    iters = [0, 7]  # number of erosion and dilation iterations for mouse mask

    # -----------------------
    # Set analysis parameters
    # -----------------------
    use_norm_to_analyze = True
    reduce_glare = True
    undistort = True
    shelter = True
    ignore_reflections = False

    # ------------------------------------------
    # Set mouse orientation detection parameters
    # ------------------------------------------
    wispy_thresh = 1.3
    wispy_erosions = 8
    speed_thresh = 4.5
    width_thresh = 1.4

    if save_videos:
        write_normal_video = False
        write_normalized_video = True
        write_cropped_mouse_video = False

# load saved settings
else:
    settings = np.load(save_folder_location + session_name_tag + "\\" + session_name_tag + "_proprocess_settings.npy")
    kernel = np.zeros(2).astype(int); iters = np.zeros(2).astype(int)
    start_frame, _, _, kernel[0], kernel[1], iters[0], iters[1], _, wispy_erosions, _, \
        width_thresh, use_norm_to_analyze, reduce_glare, undistort, shelter, frame_rate_saved_video = list(settings.astype(int))
    _, stop_frame, mask_thresh, _, _, _, _, wispy_thresh, _, speed_thresh, width_thresh, _, _, _, _, _ = list(settings)





























''' -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------              Inititialization -- functions to run once per video                 -----------------------------
# --------------------------------------------------------------------------------------------------------------------------------------'''

# -----------------------
# Perform initialization
# -----------------------
if undistort:
    maps = np.load(save_folder_location + "fisheye_maps.npy")
    map1 = maps[:, :, 0:2]
    map2 = maps[:, :, 2]

if initial_run:
    # ----------------------------------------
    # get or load background subtraction image
    # ----------------------------------------
    video_file_name = video_file_location + video_name + str(videos[0]) + ".avi"
    vid = cv2.VideoCapture(video_file_name)
    if vid.get(cv2.CAP_PROP_FRAME_COUNT) < 1:
        raise Exception(video_file_name + ' not found')
    if not os.path.isdir(save_folder_location + session_name_tag):
        os.makedirs(save_folder_location + session_name_tag)
        print("saving to " + save_folder_location + session_name_tag)        
        
    background = get_background_mean(vid, None, False, False, start_frame=start_frame,
                    file_location = save_folder_location + session_name_tag + "\\" + session_name_tag, avg_over=100)[:,:,0]

    # ----------------------
    # get shelter/arena ROI
    # ----------------------
    vid = cv2.VideoCapture(video_file_name)
    ret, frame = vid.read()
    if undistort:
        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    print("select arena!"); arena_roi = cv2.selectROI(frame[:, :, 1])
    np.save(save_folder_location + session_name_tag + "\\" + session_name_tag +  "_arena_roi.npy", arena_roi)
    if shelter:
        print("select shelter!"); shelter_roi = cv2.selectROI(frame[:, :, 1])
        np.save(save_folder_location + session_name_tag + "\\" + session_name_tag + "_shelter_roi.npy", shelter_roi)
    vid.release()

else:
    background = np.load(save_folder_location + session_name_tag + "\\" + session_name_tag + "_background_mat_avg.npy")[:,:,0]
    arena_roi = np.load(save_folder_location + session_name_tag + "\\" + session_name_tag + "_arena_roi.npy")
    if shelter:
        shelter_roi = np.load(save_folder_location + session_name_tag + "\\" + session_name_tag + "_shelter_roi.npy")


'''-------------------------------------------------------------------------------------------------------------------------------------
# ------------------------                   Set up analysis                -------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------'''

# ------------------------------
# Initialize analysis parameters
# ------------------------------
# set up mouse cropping
crop_size = 150; square_size = (crop_size, crop_size); padding = True  # size of square cropped mouse data goes into

# initialize erosion/dilation kernels
kernel_er = np.ones((kernel[0], kernel[0]), np.uint8)
kernel_dil = np.ones((kernel[1], kernel[1]), np.uint8)
kernel_head = np.ones((9, 9), np.uint8)
if reduce_glare:  # dim bright patches of background
    background[background > 170] = 140
    cv2.imshow("background", background.astype(np.uint8))
    cv2.waitKey(10)

# get coordinates in pixels of shelter and arena
shelter_x_coords = [shelter_roi[0], shelter_roi[0] + shelter_roi[2]]
shelter_y_coords = [shelter_roi[1], shelter_roi[1] + shelter_roi[3]]
arena_center = [arena_roi[0] + arena_roi[2] / 2, arena_roi[1] + arena_roi[3] / 2]
arena_radius = 1.1*np.mean([arena_center[0] - arena_roi[0], arena_roi[0] + arena_roi[2] - arena_center[0], arena_center[1] - arena_roi[1],
     arena_roi[1] + arena_roi[3] - arena_center[0]])

# set up mouse orientation detection
current_orientation = -1  # 1 or -1 sets initial orientation
topright_or_botleft = 1; topright_or_botleft_prev = 1
depth_ratio = np.ones(3)
x_center = 0; y_center = 0; move_prev = 0
history_x = np.zeros(4)
history_y = np.zeros(4)
ellipse = 0; ellipse_width = 0
slope_recipr = 1; disruption = 1


# -------------------------------
# save data for further analysis
# -------------------------------
save_file_location = save_folder_location + session_name_tag + "\\" + session_name_tag
if save_data:
    fourcc_data = cv2.VideoWriter_fourcc(*"LJPG")  # LJPG for lossless, XVID for compressed
    data_file_location = save_file_location + "_data.avi"
    if os.path.isfile(data_file_location) and do_not_overwrite:
        raise Exception("File already exists")
    data_video = cv2.VideoWriter(data_file_location, fourcc_data, frame_rate_saved_video, (crop_size, crop_size), False)
    # initialize data arrays of frame, coordinate, velocity, and frames coming after a disruption (shelter or out of bounds)
    saved_frames = np.array([]); coordinates = []; velocity = []; disruptions = []
video_nums_all_frames = []; video_nums_data_frames = []
out_of_bounds = []  # 0 is in bounds, 1 is in shelter, 2 is other


# -----------------------------------
# loop over each video in the session
# -----------------------------------
for v in videos:  

    # load video
    video_file_name = video_file_location + video_name + str(v) + ".avi"
    vid = cv2.VideoCapture(video_file_name)
    print(video_file_name + " loaded")


    # get video parameters
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end_frame = min(stop_frame, vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if width == 0:
        print("Video file not found")
        break
    # initialize video
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out_of_bound = 0

# ----------------------
# Select videos to save
# ----------------------
    session_name_tag_cur = session_name_tag + '_' + str(v)
    save_file_location_cur = save_folder_location + session_name_tag + "\\" + session_name_tag_cur
    if save_videos or save_data:
        print("saving to " + save_file_location_cur)

    fourcc_videos = cv2.VideoWriter_fourcc(*"XVID")  # LJPG for lossless, XVID for compressed
    # save the videos selected to be saved above
    if save_videos:
        normal_video, normalized_video, cropped_mouse, stereo_input, stereo_input_R, threeD_combined, threeD_smooth = \
            write_videos(save_file_location_cur, save_videos, do_not_overwrite,
                         fourcc_videos, frame_rate_saved_video, width, height, crop_size, crop_size, write_normal_video,
                         write_normalized_video, write_cropped_mouse_video, False, False, False)


    """-------------------------------------------------------------------------------------------------------------------------------------
     ------------------------                   Perform analysis                ------------------------------------------------------------
     --------------------------------------------------------------------------------------------------------------------------------------"""

    while True:
        # grab the frame
        ret, frame = vid.read()
        frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_num >= end_frame-1:
            print("fin")
            break

        if ret:
            # ---------------------------------------------------------
            # Perform background subtraction and fisheye rectification
            # ---------------------------------------------------------
            frame = frame[:, :, 1]
            frame_norm = (frame / background)
            video_nums_all_frames.append(v)
            if undistort:
                frame_norm = cv2.remap(frame_norm, map1, map2, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=255)

            # ----------------------------------
            # Find the contour around the mouse
            # ----------------------------------
            # use the thresholds, erosion, and dilation set above to extract a mask coinciding with the mouse
            frame_norm_mask = (frame_norm < mask_thresh).astype(np.uint8)
            frame_norm_mask = cv2.erode(frame_norm_mask, kernel_er, iterations=iters[0])
            frame_norm_mask = cv2.dilate(frame_norm_mask, kernel_dil, iterations=iters[1])

            x_center_prev = x_center; y_center_prev = y_center   #  save the previous center of mouse in order to extract velocity
            try:  # extract the largest contour in this mask -- this should correspond to the mouse
                contours, big_cnt_ind, x_center, y_center, _ = get_biggest_contour(frame_norm_mask)
                prev_ellipse = ellipse
                ellipse = cv2.fitEllipse(contours[big_cnt_ind])
                ellipse_width = (ellipse[1][1] / ellipse[1][0])

            # --------------------------------------------------------------------------------
            # If the mouse is obscured or in the shelter, updata data arrays and skip analysis
            # --------------------------------------------------------------------------------
#         **** this section is specific to the Branco Lab Barns Maze -- modify for other arenas ******
            except:
                disruption = 1 # The velocity of the saved frame following this is invalid
                if y_center < max(shelter_y_coords) + 20:  # if in shelter (specific to Barns Maze setup)
                    if out_of_bound != 1:
                        print("out of bounds, sheltered")
                    out_of_bound = 1
                else:  # out of bounds but not in shelter
                    out_of_bound = 2
                out_of_bounds.append(out_of_bound)
                continue

            # if there is a shadow mouse appearing outside of the arena:
            if np.sqrt((y_center - arena_center[1]) ** 2 + (x_center - arena_center[0]) ** 2) > arena_radius and \
                    np.sqrt((y_center - y_center_prev) ** 2 + (x_center - x_center_prev) ** 2) > (speed_thresh * 3) and ignore_reflections:
                try:  # not the real mouse but a shadow of a mouse occurring outside the boundaries :o
                    if shadow != 1:
                        print("beware! a shadow mouse approaches...")
                        shadow = 1
                    big_cnt_ind, x_center, y_center, _, _ = get_second_biggest_contour(frame_norm_mask, 0, 0)
                    ellipse = cv2.fitEllipse(contours[big_cnt_ind])  # so get the second biggest contour, corresponding to the mouse
                    ellipse_width = (ellipse[1][1] / ellipse[1][0])
                except: # if, while there's a shadow mouse, the actual mouse happens to be out of bounds:
                    if y_center_prev < max(shelter_y_coords) + 20:
                        out_of_bound = 1
                    else:
                        out_of_bound = 2
                    out_of_bounds.append(out_of_bound)
                    continue
            else:
                shadow = 0

            # if someone's arm comes into the arena
            if np.sum(frame_norm_mask) > 30000:
                print("what are you doing there?")
                print(np.sum(frame_norm_mask))
                disruption = 1
                out_of_bound = 2
                out_of_bounds.append(out_of_bound)
                continue

            # If mouse is sheltered, note that in the out of bounds/disruption arrays and continue
            if min(shelter_x_coords) <= x_center <= max(shelter_x_coords) and \
                    min(shelter_y_coords) <= y_center <= max(shelter_y_coords):
                disruption = 1
                if out_of_bound != 1:
                    print("sheltered...")
                out_of_bound = 1
                out_of_bounds.append(out_of_bound)
                if show_images:
                    cv2.imshow("2D_norm", frame_norm)
                    if cv2.waitKey(int(1000 / frame_rate_online_display)) & 0xFF == ord("q"):
                        break
                continue
            else:
                out_of_bound = 0
                # flip mouse, if it is elliptical and walking out of shelter
                if np.mean(out_of_bounds[-200:]) ==1 and ellipse_width > width_thresh:  
                    current_orientation *= -1
#          ******** arena-specific section ends here *********

            # --------------------------
            # Apply mask and crop mouse
            # --------------------------
            # Pad image if mouse is near the edge of the image
            if abs(x_center - width) <= crop_size or abs(y_center - height) <= crop_size:
                padding = True
                border_size = crop_size
            else:
                padding = False
                border_size = 0

            # create a new mask, corresponding to only the largest contour
            blank = np.zeros(frame_norm.shape).astype(np.uint8)
            contour_mask = cv2.drawContours(blank, contours, big_cnt_ind, color=(1, 1, 1), thickness=cv2.FILLED)

            # apply this mask to the original image
            if use_norm_to_analyze:
                frame_masked = frame_norm * contour_mask * 128
            else:
                if undistort:
                    frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=255)
                frame_masked = frame * contour_mask

            # crop the original, masked image to the crop size
            if padding:
                frame_masked = cv2.copyMakeBorder(frame_masked, border_size, border_size, border_size, border_size,
                                                  cv2.BORDER_CONSTANT, value=0)
            frame_masked_cropped = (frame_masked[y_center - int(crop_size / 2) + border_size:y_center + int(crop_size / 2) + border_size,
                                                 x_center - int(crop_size / 2) + border_size:x_center + int(crop_size / 2) + border_size]).astype(np.uint8)

            # ------------------------------
            # Get spine aligned orientation
            # ------------------------------
            # flip mouse into the correct orientation
            try:
                rotate_angle, current_orientation, prev_ellipse, topright_or_botleft, ellipse_width = \
                    flip_mouse(current_orientation, ellipse, topright_or_botleft, frame_masked_cropped, width_thresh=width_thresh)
            except:
                print("mouse out of bounds but not captured above...")
                out_of_bound = 2
                out_of_bounds.append(out_of_bound)
                disruption = 1
                continue

            # Rotate mouse to make it straight
            M = cv2.getRotationMatrix2D((int(crop_size / 2), int(crop_size / 2)), rotate_angle, 1)
            frame_masked_straight = cv2.warpAffine(frame_masked_cropped, M, (crop_size, crop_size))

            # check for errors -- if the tail end is less tumescent, or the mouse is running toward its tail, flip mouse 180 degrees
            frame_eroded = frame_masked_straight.copy(); frame_eroded[frame_eroded > 100] = 0
            frame_eroded = cv2.erode(frame_masked_straight, kernel_er, iterations=wispy_erosions)
            pixels_top = frame_eroded[0:int(crop_size / 2) + kernel[1], :]; pixels_bottom = frame_eroded[int(crop_size / 2) + kernel[1]:]

            rotate_angle, current_orientation, depth_ratio, history_x, history_y, head_dir_x_component, head_dir_y_component, flip = \
                correct_flip(frame_num - start_frame, current_orientation, pixels_top, pixels_bottom, history_x, history_y,
                             x_center, y_center, ellipse, ellipse_width, \
                             width_thresh=width_thresh, speed_thresh=speed_thresh, wispy_thresh=wispy_thresh)
            if flip:
                print("frame " + str(frame_num - start_frame))
            M = cv2.getRotationMatrix2D((int(crop_size / 2), int(crop_size / 2)), rotate_angle, 1)

            # straighten data images after taking flip into account
            frame_masked_straight = cv2.warpAffine(frame_masked_cropped, M, square_size)

            # -------------
            # Get velocity
            # -------------
            # use center of mouse and orientation to get get mouse velocity relative to orientation
            delta_x = x_center - x_center_prev
            delta_y = - y_center + y_center_prev

            head_dir_vector_length = np.sqrt(head_dir_x_component ** 2 + head_dir_y_component ** 2)
            head_dir = [head_dir_x_component / head_dir_vector_length, -head_dir_y_component / head_dir_vector_length]
            head_dir_ortho = [-head_dir_y_component / head_dir_vector_length, -head_dir_x_component / head_dir_vector_length]
            vel_along_head_dir = np.dot([delta_x, delta_y], head_dir)
            vel_ortho_head_dir = np.dot([delta_x, delta_y], head_dir_ortho)

            # ---------------------
            # Save and display data
            # ---------------------      
            # save data for further analysis
            if save_data:
                data_video.write(frame_masked_straight)
                velocity.append([vel_along_head_dir, vel_ortho_head_dir, head_dir[0], head_dir[1]])
                saved_frames = np.append(saved_frames, frame_num - 1)  # minus 1 to put in python coordinates
                video_nums_data_frames.append(v)
                coordinates.append([x_center, y_center])
                disruptions.append(disruption)
                out_of_bounds.append(out_of_bound)
                disruption = 0

            # save videos
            if save_videos:
                if write_normal_video:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    normal_video.write(frame)
                if write_normalized_video:
                    normalized_video.write(cv2.cvtColor((128 * frame_norm).astype(np.uint8), cv2.COLOR_GRAY2RGB))
                if write_cropped_mouse_video:
                    frame_masked_straight_resized = cv2.resize(frame_masked_straight, (crop_size * 3, crop_size * 3))
                    frame_masked_straight_resized = cv2.cvtColor(frame_masked_straight_resized, cv2.COLOR_GRAY2RGB)
                    cropped_mouse.write(frame_masked_straight_resized)

            # display videos
            if show_images:
                frame_masked_straight_resized = cv2.resize(frame_masked_straight, (crop_size * 3, crop_size * 3))
                # cv2.imshow("2D", frame[:,:,l])
                cv2.imshow("2D_norm", frame_norm)
                # cv2.imshow("2D_norm_cropped", frame_masked_cropped)
                cv2.imshow("2D straight", frame_masked_straight_resized)
            if cv2.waitKey(int(1000 / frame_rate_online_display)) & 0xFF == ord('q'):
                print(vid.get(cv2.CAP_PROP_POS_FRAMES))
                break

            # display notification every 500 frames
            if (frame_num - start_frame) % 500 == 0:
                print(str(int(frame_num - start_frame)) + " out of " + str(
                    int(end_frame - start_frame)) + " frames complete")

        else:
            print("frame-grabbing problem")
            cv2.waitKey(100)


    """--------------------------------------------------------------------------------------------------------------
    ------------------                   Wrap up and save data              ---------------------------------------
    # -----------------------------------------------------------------------------------------------------------"""
    # end videos being read and saved
    vid.release()
    if save_videos:
        if write_normal_video:
            normal_video.release()
        if write_normalized_video:
            normalized_video.release()
        if write_cropped_mouse_video:
            cropped_mouse.release()

# save position, velocity, and frame numbers
if save_data:
    data_video.release()

    #make folder to save preprocessing results
    if not os.path.isdir(save_folder_location + session_name_tag):
        os.makedirs(save_folder_location + session_name_tag)

    try:
        velocity[0][0:2] = [0, 0]
    except:
        print("session spent in shelter")   # <-- if session was only spent in shelter
    saved_frames_plus_video_num = np.zeros((len(saved_frames),2))
    out_of_bounds_plus_video_num = np.zeros((len(out_of_bounds),2))
    saved_frames_plus_video_num[:,0] = saved_frames - start_frame
    saved_frames_plus_video_num[:, 1] = video_nums_data_frames
    out_of_bounds_plus_video_num[:, 0] = out_of_bounds
    out_of_bounds_plus_video_num[:, 1] = video_nums_all_frames
    np.save(save_file_location + "_coordinates.npy", coordinates)
    np.save(save_file_location + "_velocity.npy", velocity)
    np.save(save_file_location + "_frames", saved_frames_plus_video_num)
    np.save(save_file_location + "_disruption", disruptions)
    np.save(save_file_location + "_out_of_bounds", out_of_bounds_plus_video_num)

    # save analysis settings
    settings = [start_frame, stop_frame, mask_thresh, kernel[0], kernel[1], iters[0], iters[1], wispy_thresh, wispy_erosions,
                speed_thresh, width_thresh, use_norm_to_analyze, reduce_glare, undistort, shelter, frame_rate_saved_video]
    np.save(save_file_location + "_proprocess_settings", np.array(settings))



cv2.destroyAllWindows()