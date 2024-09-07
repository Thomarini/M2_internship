import os
import numpy as np
import matplotlib.pyplot as plt



# ==================

from utils.helper1 import *
from utils.helper2 import *

os.chdir("C:/Users/Thomas/Documents/stage/edge_consistency_v1/output_files/tmp")


spine_lengths = []
e_s_rad = []
s_rad = []

spine_median_length = 210 #np.median(results[:, 0], axis=0)

for frame_nb in range(0, 1000):
    try:
        spine = read_file("extended_spine", frame_nb)
        skinPoints = read_file("yolo_edge", frame_nb)
        spine_length = getSpineLength(spine)
        spine_lengths.append(spine_length)
        asymetric_dist = get_dif_dist_spine_skin(spine, skinPoints, nb_point_per_spine = 25)

        spines_length_dif = (spine_length - spine_median_length)**2
        hyperparameter = 1000
        criteria =  asymetric_dist + hyperparameter*spines_length_dif
        s_rad.append(asymetric_dist)
        e_s_rad.append(criteria)
    except: 
        pass
    
plt.plot(spine_lengths, e_s_rad, marker="*", color = "blue", ms = 12, linestyle = "")
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.xlabel("spine length [px]", fontsize = 15)
plt.ylabel("extended S-RAD [px^2]", fontsize = 15)
plt.show()

plt.plot(spine_lengths, s_rad, marker="*", color = "red", ms = 12, linestyle = "")
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.xlabel("spine length [px]", fontsize = 15)
plt.ylabel("S-RAD [px]", fontsize = 15)
plt.show()
