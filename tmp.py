import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from utils.helper1 import read_file, load_frames_spines_from_csv, display_shape, display_spine, display_fig
from utils.helper2 import *
from shape_processing import get_estimated_shape
from spine_postprocessing import *
import cv2


os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\tmp") 
skin_points = read_file("applied_shape", 429)
spine = read_file("normalized_spine", 429)
display_shape(skin_points)
display_spine(spine, unicolor="gold", marker = ".")
plt.plot([spine[-1][0], spine[-10][0]], [spine[-1][1], spine[-10][1]], linestyle = "-", linewidth = 4, color = "purple")
plt.plot(spine[-1][0], spine[-1][1], "ro", ms = 16, linewidth=4)
plt.plot( spine[-10][0], spine[-10][1], "bo", ms = 16, linewidth=4)
plt.plot(spine[0][0], spine[0][1], "go", ms = 16, linewidth=4)



display_fig()
quit()

""" skin_points = read_file("yolo_edge", 427)
spine = read_file("extended_spine", 427)

display_shape(skin_points)
display_spine(spine)

shape_width, desired_length = read_file("ref_width", 391), read_file("ref_length", 391)
#spine = fit_single_spine_IoU(skinPoints, spine, frame, desired_length, shape_width)
_, applied_shape = get_estimated_shape(spine, shape_width)

spine = np.array(direction_uniformization(spine, skin_points))
new_spine = overextend_single_spine_to_skin(spine, skin_points, 427)
display_spine(new_spine, unicolor="red")
resampled_spine = resample_spine(new_spine, 1000)

cpt = 0
cpt_maxi = -1
maxi = 0
while new_spine is not None:
    new_spine = get_spine_candidate(resampled_spine, cpt, desired_length)
    if new_spine is None:
        break
    
    else:
        # The method cannot produce valid spine anymore
        if (getSpineLength(new_spine) < desired_length*0.8):
            break

        #display_discret_spine(spine, "D", unicolor = colors[cpt])
        _, applied_shape = get_estimated_shape(spine, shape_width)
        IoU = calculate_iou(applied_shape, skin_points, new_spine, 427)
        #print(cpt, IoU)
        if (IoU > maxi):
            maxi = IoU
            cpt_maxi = cpt
        if (IoU == -np.inf):
            pass
    cpt += 1
    
    
    if cpt == 500:
        break
    
if cpt_maxi != -1:
    new_spine = get_spine_candidate(resampled_spine, cpt_maxi, desired_length)
    _, applied_shape = get_estimated_shape(new_spine, shape_width)
    new_spine = resample_spine(new_spine, 50)  
    display_spine(new_spine, unicolor="dimgrey")
    _, applied_shape = get_estimated_shape(new_spine, shape_width)
    display_shape(applied_shape, color="lightblue", ms = 15) """




os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\final") 
display_fig()
#plt.savefig("yolo_512.png")

