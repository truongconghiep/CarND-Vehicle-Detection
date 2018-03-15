'''
Created on 10.03.2018

@author: Hiep Truong
'''
import numpy as np
import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
from Finding_Car import Find_Car_Multi_Scale, Window_buf
from Training_Model import Training_Classifier_Pipeline, ReadSvcFromPickle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from lesson_functions import add_heat, apply_threshold, draw_labeled_bboxes

def Find_Car_In_Frame(img):
    global svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, ystart, ystop, scales, colorspace, buffer, heatmap_threshold
    out_rectangles = Find_Car_Multi_Scale(scales, img, colorspace, 
                                          ystart, ystop, 
                                          svc, X_scaler, 
                                          orient, 
                                          pix_per_cell, cell_per_block, 
                                          spatial_size, hist_bins)
    buffer.push_wins(out_rectangles)
    rectang = buffer.get_concate()
    heat_map = add_heat(img, rectang)
    heatmap_img = apply_threshold(heat_map, heatmap_threshold)
    labels = label(heatmap_img)
    draw_img, rect = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img

def Processing_video(input_video, output_video):
    clip1 = VideoFileClip(input_video)
    white_clip = clip1.fl_image(Find_Car_In_Frame)
    white_clip.write_videofile(output_video, audio=False)

dirName ='./FeatureSetBig'
# Classifier = Training_Classifier_Pipeline(dirName, color_space = 'YUV',
# #                                            orient = 11, pix_per_cell=16,
#                                             spatial_size=(16,16))

img = mpimg.imread('./test_images/test1.jpg')
svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins = ReadSvcFromPickle('Svm.pkl')
ystart = 300
ystop = 700
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
scales = [1.5,2.0, 2.3, 2,7, 3.0,3.5]
heatmap_threshold = 5
buffer = Window_buf(10)

input_project_video = './test_videos/challenge.mp4'
file_name =  input_project_video.replace('./test_videos/', '')
output_project_video = 'output_' + file_name


Processing_video(input_project_video, output_project_video)

