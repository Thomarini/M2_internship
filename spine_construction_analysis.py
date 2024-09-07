import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from utils.helper1 import read_file, load_frames_spines_from_csv, display_shape, display_spine, display_fig
from utils.helper2 import *
from shape_processing import get_estimated_shape
from spine_postprocessing import filtering_single_spine
import cv2





os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\final")

start_ToI = 2*60 + 12 #1*60 + 35#13*60+10 #2*60+15 # 13*60+48 #
end_ToI = 3*60 + 30 #1*60 + 56#13*60+24 # 14*60 + 2 #
vid_name = "000000.mp4" + "{}".format(0)
fps = 25



# Load the background image
os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\final")
background = cv2.imread('background.jpg')
height, width, _ = background.shape

path_to_vid = r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\input_files\000000.mp4"



vidcap = cv2.VideoCapture(path_to_vid)
if vidcap.isOpened() is False:
    print("Can't load the input video")
    quit()
    
vidcap.set(cv2.CAP_PROP_POS_MSEC, start_ToI*1e3)     






# ====================






os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\final")
frames_file, spines = load_frames_spines_from_csv("spines_interpolated")
trajectory_data = spines
num_frames = np.shape(spines)[0] 
num_trajectories = np.shape(spines)[1]      

# Assign a unique color to each trajectory
colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_trajectories)]

# Draw trajectory points on each frame
frames = []
idx = 0
missing_index = 0
offset = 28
for _ in range(offset):
    ret, frame = vidcap.read()

missing = 0
for frame_idx in range(4, num_frames):
    # Copy the background image
    #frame = background.copy()
    
    ret, frame = vidcap.read()
    
    try:
        os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\tmp")     
        skin_points = read_file("yolo_edge", frame_idx+offset-3)  
        #cv2.polylines(frame, np.int32([skin_points]), True, [0, 131, 255], 2)
        idx += 1
    except:
        pass
    try:
        #cv2.polylines(frame, np.int32([spine]), False, [255, 255, 0], 2)
        #spine = read_file("extended_spine", frame_idx+offset)
        #cv2.polylines(frame, np.int32([spine]), False, [0, 0, 255], 3)     
        skin_points = read_file("yolo_edge", frame_idx+offset-3)  
        #spine = trajectory_data[frame_idx-4-missing_frame, :, :]
        
        #spine = read_file("normalized_spine", frame_idx+offset-3)  
        
        os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\tmp") 
        shape_width, _ = read_file("ref_width", 391), read_file("ref_length", 391)
        spine = trajectory_data[frame_idx-4, :, :]
        _, applied_shape = get_estimated_shape(spine, shape_width)
        cv2.polylines(frame, np.int32([applied_shape]), True, [255, 255, 0], 2)

        #cv2.polylines(frame, np.int32([trajectory_data[frame_idx-4, :, :]]), False, [51, 255, 255], 2)        
        #cv2.polylines(frame, np.int32([spine]), False, [51, 255, 255], 2)        
        #display_shape(skin_shape)
        
        # Draw each trajectory point
        for color_index, traj_idx in enumerate([-1, 0, -10]):
            #point = tuple(map(int, trajectory_data[frame_idx-4-missing_frame, traj_idx, :]))
            point = tuple(map(int, spine[traj_idx]))
            color = [[0, 0, 255], [0, 255, 0], [255, 0, 0]] # raw shape color
            #color = [0, 100, 0] # interpolated color
            #cv2.circle(frame, point, radius=2, color=color[color_index], thickness=2)
            
        
        os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\final\tmp")
        cv2.imwrite("shape_yolo_PoI{}.png".format(frame_idx), frame)
    except Exception as e:
        os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\final\tmp")
        cv2.imwrite("shape_yolo_PoI{}.png".format(frame_idx), frame)
    
    
    
    
    # Add the frame to the list
    frames.append(frame)


vidcap.release()

# Compile frames into a video
os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\final")
out = cv2.VideoWriter('final.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))

for frame in frames:
    out.write(frame)

out.release()

quit()




#===============================================

os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\tmp")

frame = [34, 37, 41, 44, 49, 67, 68, 144, 158, 179, 184, 192, 193, 198, 199, 203, 204, 205, 206, 207, 209, 214, 224, 226, 227, 230, 234, 235, 236, 237, 239, 242, 249, 247, 250, 272, 278, 279, 280, 282, 283, 289, 291, 325, 380, 392, 398, 401, 422, 424, 425, 426, 427, 430, 431, 433, 434, 437, 439, 440, 454, 489, 508, 511, 551, 618]
print(len(frame))

missing_frame = []
missing_spine_generation = []
for frame_idx in range(34, 799):
    try:
        os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\tmp")
        skin_points = read_file("yolo_edge", frame_idx)         
    
    except:
        missing_frame.append(frame_idx)
        
    try:
        os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\tmp")
        spine = read_file("applied_shape", frame_idx)
        if spine is None:
            missing_spine_generation.append(frame_idx)
    except:
        pass

print(missing_frame)
print(missing_spine_generation)
missing_spine_generation = [element for element in missing_spine_generation if element not in missing_frame]
n_bins = 20

# Create histogram
plt.figure(figsize=(10, 6))
#plt.hist([frame, missing_frame, missing_spine_generation], bins=n_bins, edgecolor='black', color = ["blue", "red", "green"], density=False, histtype='barstacked')
plt.hist(missing_frame, bins=n_bins, color= "red", edgecolor='black')
plt.hist(frame, bins=n_bins, color= "blue", edgecolor='black')
plt.hist(missing_spine_generation, bins=n_bins, color= "green", edgecolor='black')

# Set title and labels
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.xlabel('Frame number', fontsize=18)
plt.ylabel('Number of frame', fontsize=18)
#plt.legend(["Missing frames", "Absurd shape"], fontsize=18)
plt.legend(["Missing frames", "Absurd shape", "Frame lost"], fontsize=15)

# Show the plot
plt.grid(True)
plt.show()

quit()




for frame in range(0, 799):
    try:
        os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\tmp")
        skin_points = read_file("yolo_edge", frame)         
        spine = read_file("spine_estimation", frame) 
        display_shape(skin_points)
        display_spine(spine)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\final\yolo_spine")
        plt.savefig("spine_yolo{}".format(frame))
        plt.clf()
    except:
        print(frame)


quit()


# Load the background image
os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\final")
background = cv2.imread('background.jpg')
height, width, _ = background.shape

path_to_vid = r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\input_files\000000.mp4"



vidcap = cv2.VideoCapture(path_to_vid)
if vidcap.isOpened() is False:
    print("Can't load the input video")
    quit()
    
vidcap.set(cv2.CAP_PROP_POS_MSEC, start_ToI*1e3)     

frames, spines = load_frames_spines_from_csv("spines_interpolated_0")
trajectory_data = spines
num_frames = np.shape(spines)[0] 
num_trajectories = np.shape(spines)[1]      

# Assign a unique color to each trajectory
colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_trajectories)]

# Draw trajectory points on each frame
frames = []
missing_frame = 0
offset = 28 +2
for _ in range(offset):
    ret, frame = vidcap.read()


for frame_idx in range(num_frames):
    # Copy the background image
    #frame = background.copy()
    
    ret, frame = vidcap.read()
    if frame_idx == 7:
        pass 
    # Draw each trajectory point
    for color_index, traj_idx in enumerate([-1, 0, -10]):
        point = tuple(map(int, trajectory_data[frame_idx, traj_idx, :]))
        color = [[0, 0, 255], [0, 255, 0], [255, 0, 0]] # raw shape color
        #color = [0, 100, 0] # interpolated color
        #cv2.circle(frame, point, radius=2, color=color[color_index], thickness=2)
    

    try:
        os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\tmp")     
        skin_points = read_file("yolo_edge", frame_idx+offset)         
        spine = read_file("spine_estimation", frame_idx+offset) 
        print(frame_idx)  
        spine = filtering_single_spine(skin_points, spine, frame, 1)      
        cv2.polylines(frame, np.int32([skin_points]), True, [0, 131, 255], 2)
        cv2.polylines(frame, np.int32([spine]), False, [100, 210, 0], 2)
        #display_shape(skin_shape)
        os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\final")
        cv2.imwrite("yolo_edge_{}.png".format(frame_idx), frame)
    except:
        cv2.imwrite("yolo_edge_{}.png".format(frame_idx), frame)
        
    # Add the frame to the list
    frames.append(frame)


vidcap.release()

# Compile frames into a video
out = cv2.VideoWriter('output{}.avi'.format(50), cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))

for frame in frames:
    out.write(frame)

out.release()