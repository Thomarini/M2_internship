import os
import matplotlib.pyplot as plt
import numpy as np

from utils.helper1 import *
from utils.helper2 import *

from shape_processing import get_estimated_shape


os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\final")

def display_spine_length():

    spine_length = []

    for spine_idx in range(37, 192):
        frames, spine = load_frames_spines_from_csv("spines_{}".format(spine_idx))
        spine_length.append(getSpineLength(spine[0]))
        

    cpt = 0
    for length in spine_length:
        if 210*1.05 > length > 210*0.95:
            cpt += 1

    print(cpt/len(spine_length)*100, "%")

    plt.plot(np.arange(0, len(spine_length)), spine_length, marker="*", color = "red", ms = 12, linestyle = "")
    plt.plot(np.arange(0, len(spine_length)), np.ones(len(spine_length))*210, color = "green", linewidth=2, linestyle = "-")
    plt.plot(np.arange(0, len(spine_length)), np.ones(len(spine_length))*210*0.95, color = "black", linewidth=2, linestyle = "--")
    plt.plot(np.arange(0, len(spine_length)), np.ones(len(spine_length))*210*1.05, color = "black", linewidth=2, linestyle = "--")

    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    plt.xlabel("morphology generation", fontsize = 15)
    plt.ylabel("spine length [px]", fontsize = 15)

    plt.legend(["morphology spine length", "desired spine length", "acceptable range +- 5%"], fontsize = 15)
    plt.show()

def display_fish_shape():
    os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\tmp")
    
    frames = get_relevant_frames("ref_length")
    
    for frame in frames:
        dist_spine_skin = np.load("ref_width{}.npy".format(frame))
        spine_length = np.load("ref_length{}.npy".format(frame))
        spine = np.linspace([0, 0], [spine_length, 0], 50)
        spine = np.flip(spine, axis = 0)
        _, estimated_shape = get_estimated_shape(spine, dist_spine_skin)
        estimated_shape = np.vstack([estimated_shape, estimated_shape[0, :]])
        plt.plot(estimated_shape[:, 0], estimated_shape[:, 1], marker="", ms = 12, linestyle = "-")
    
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.show()

def display_length_variation_after_interpolation():
    os.chdir(r"C:\Users\Thomas\Documents\stage\edge_consistency_v1\output_files\final")
    
    frames, spine = load_frames_spines_from_csv("spines_interpolated_82")
    
    spine_length = []
    for frame_idx, frame in enumerate(frames):
        spine_length.append(getSpineLength(spine[frame]))
    
    plt.plot(np.arange(0, len(spine_length)), spine_length, marker="*", color = "red", ms = 12, linestyle = "")
    plt.plot(np.arange(0, len(spine_length)), np.ones(len(spine_length))*np.mean(spine_length), color = "green", linewidth=2, linestyle = "-")
    plt.plot(np.arange(0, len(spine_length)), np.ones(len(spine_length))*np.mean(spine_length)*0.98, color = "black", linewidth=2, linestyle = "--")
    plt.plot(np.arange(0, len(spine_length)), np.ones(len(spine_length))*np.mean(spine_length)*1.02, color = "black", linewidth=2, linestyle = "--")

    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    plt.xlabel("frame number", fontsize = 15)
    plt.ylabel("spine length [px]", fontsize = 15)

    plt.legend(["spine length at given frame", "average spine length", "average spine length +- 2%"], fontsize = 15)
    plt.show()
    

display_length_variation_after_interpolation()