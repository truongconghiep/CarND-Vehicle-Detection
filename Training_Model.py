'''
Created on 10.03.2018

@author: Hiep Truong
'''
import time
import glob
from lesson_functions import extract_features
import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

def Read_Data(dirName):
    # Read in car and non-car images
    images = glob.glob(dirName + '/*.png')
    cars = []
    notcars = []
    for image in images:
        if 'image0' in image:
            cars.append(image)
        elif 'image' in image or 'extra' in image:
            notcars.append(image)
        else:
            cars.append(image)
    return cars, notcars
    

def SaveToPickle(name, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    SaveToPickle = {"svc": svc, "scaler": X_scaler, "orient": orient, "pix_per_cell": pix_per_cell, "cell_per_block": cell_per_block, "spatial_size": spatial_size, 
                    "hist_bins":hist_bins}
    Svm_pkl = name
# Open the file to save as pkl file
    Svm_pkl = open(Svm_pkl, 'wb')
    pickle.dump(SaveToPickle, Svm_pkl)
# Close the pickle instances
    Svm_pkl.close()
    
def ReadSvcFromPickle(name):
    dist_pickle = pickle.load( open(name, "rb" ) )
    svc = dist_pickle["svc"]
    print(svc)
    X_scaler = dist_pickle["scaler"]
    print(X_scaler)
    orient = dist_pickle["orient"]
    print("orient ", orient)
    pix_per_cell = dist_pickle["pix_per_cell"]
    print("pix_per_cell ",pix_per_cell)
    cell_per_block = dist_pickle["cell_per_block"]
    print("cell_per_block ",cell_per_block)
    spatial_size = dist_pickle["spatial_size"]
    print("spatial_size ", spatial_size)
    hist_bins = dist_pickle["hist_bins"]
    print("hist_bins ", hist_bins)
    return svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins

def Training_Classifier_Pipeline(data, output_name = 'Svm.pkl', color_space = 'YUV', 
                                 spatial_size=(32,32), hist_bins=32, orient = 9, 
                                 pix_per_cell = 8, cell_per_block = 2, hog_channel = "ALL",
                                 spatial_feat=True, hist_feat=True, hog_feat=True):
    # Read data
    cars, notcars = Read_Data(data)
    # Reduce the sample size because HOG features are slow to compute
    # The quiz evaluator times out after 13s of CPU time
#     sample_size = 5000
#     cars = cars[0:sample_size]
#     notcars = notcars[0:sample_size]
    # Get feature in data images
    t=time.time()
    car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)
        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)


    print('Using:',orient,'orientations',pix_per_cell,
          'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    # Test the model
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    # save model
    SaveToPickle(output_name, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    return output_name