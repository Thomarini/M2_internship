import cv2

from utils.helper1 import *
from utils.helper2 import *





os.chdir("C:/Users/Thomas/Documents/stage/edge_consistency_v1")

# Video path
path_to_vid = "input_files/000000.mp4"

# Generated data path
temporary_files_path = r"C:/Users/Thomas/Documents/stage/edge_consistency_v1/output_files/tmp/"
final_output_path = r"C:/Users/Thomas/Documents/stage/edge_consistency_v1/output_files/final/"

start_ToI = 2*60 + 12 #1*60 + 35#13*60+10 #2*60+15 # 13*60+48 #
end_ToI = 2*60 + 48 #1*60 + 56#13*60+24 # 14*60 + 2 #

# Get video properties
vidcap = cv2.VideoCapture(path_to_vid)
vidcap.set(cv2.CAP_PROP_POS_MSEC, start_ToI*1e3) 
fps = vidcap.get(cv2.CAP_PROP_FPS)
frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nb_FoI = int((end_ToI - start_ToI) * fps)

# Define codec and create VideoWriter for the output video
os.chdir(final_output_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (frame_width, frame_height))

count = 0
while vidcap.isOpened() and count < nb_FoI:
    ret, frame = vidcap.read()
    if not ret:
        break
    
    os.chdir(temporary_files_path)
    
    try:
        yolo_output = read_file("yolo_edge", count)
        for point in yolo_output:
            cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)  # Green circles
    except:
        pass
    try:
        applied_shape = read_file("applied_shape", count)
        for point in applied_shape:
            cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)  # Red circles
        
    except:
        pass
    
    os.chdir(final_output_path)
    frame_path = os.path.join("frames", f"frame_{count:04d}.png")
    cv2.imwrite(frame_path, frame)
    
    # Write the frame to the output video
    out.write(frame)

    count += 1