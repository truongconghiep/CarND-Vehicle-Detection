'''
Created on 11.03.2018

@author: Hiep Truong
'''
import numpy as np

from collections import deque
from lesson_functions import find_cars

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
    
def Find_Car_Multi_Scale(scales, img, colorspace, ystart, ystop, 
                         svc, X_scaler, orient, 
                         pix_per_cell, cell_per_block, 
                         spatial_size, hist_bins):
    out_rectagles = []
    for scale in scales:
        out_rectagles = out_rectagles + (find_cars(img, colorspace, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    return out_rectagles