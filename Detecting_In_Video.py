'''
Created on 10.03.2018

@author: Hiep Truong
'''
import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
import numpy as np
from Training_Model import Training_Classifier_Pipeline, ReadSvcFromPickle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from collections import deque
from lesson_functions import find_cars, add_heat, apply_threshold, draw_labeled_bboxes

dirName ='C:/Users/Hiep Truong/Desktop/CarND_projects/Vehicle-detection-and-tracking/FeatureSetBig'
# Classifier = Training_Classifier_Pipeline(dirName, color_space = 'YUV',
# #                                           , orient = 11, pix_per_cell=16,
#                                             spatial_size=(16,16)
#                                         )


def Rotate_List_Left(List):
    a = deque(List)
    a.rotate()
    return a

class Window_buf:
    def __init__(self, sz):
        self.size = sz # The size of the buffer
        self.num = 0 # The number of frames stored in the buffer
        self.buf = []
    # push detected windows to the buffer
    def push_wins(self, wins):
        if wins is not None:
            if self.num < self.size:
                self.buf.append(wins)
                self.num += 1
            else:
                self.buf = Rotate_List_Left(self.buf)
                self.buf[-1] = wins
        else:
            if self.num != 0:
                self.num -=1
    # Concatenate buffered windows
    def get_concate(self):
        result = []
        if self.num > 0:
            result = np.concatenate(self.buf)
        return result
    
def Find_Car_Multi_Scale(scales, img, colorspace, ystart, ystop, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    out_rectagles = []
    for scale in scales:
        out_rectagles = out_rectagles + (find_cars(img, colorspace, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    return out_rectagles

def Find_Car_In_Frame(img):
    global svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, ystart, ystop, scale, colorspace, buffer, heatmap_threshold
    out_rectangles = Find_Car_Multi_Scale(scales, img, colorspace, ystart, ystop, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
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

img = mpimg.imread('./test_images/test1.jpg')
svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins = ReadSvcFromPickle('Svm.pkl')
ystart = 300
ystop = 700
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
scales = [1.5,2.0, 2.3, 2,7, 3.0,3.5]
heatmap_threshold = 3
buffer = Window_buf(7)

input_project_video = 'project_video.mp4'
file_name =  input_project_video.replace('./test_videos/', '')
output_project_video = 'output_' + file_name

Processing_video(input_project_video, output_project_video)

