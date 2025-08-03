import os
import cv2
import numpy as np
import joblib
from skimage import io, color, feature

# Load the trained model
ker = 'poly'  # kernel type, change to 'linear', 'rbf', 'sigmoid' or 'poly' for different kernels
deg = 3
gam = 0.1
c = 75
coef = 1
date = '02082025'   #ddmmyyyy: date of the model
time = '104313'   #hhmmss: time of the model

model = joblib.load(f'../models/kernel-{ker}_deg-{deg}_gamma-{gam}_C-{c}_coef0-{coef}_{date}_{time}.pkl')

# Define prediction folder
prediction_folder = '/prediction'

# Preprocess and predict
for file in os.listdir(prediction_folder):
    file_path = os.path.join(prediction_folder, file)

    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # Load and preprocess image
        image = io.imread(file_path)
        if len(image.shape) > 2 and image.shape[2] == 3:
            image = color.rgb2gray(image)
        image = cv2.resize(image, (200, 200))

        # Extract HOG features
        hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

        hog_features = hog_features.reshape(1, -1)

        # Predict
        pred = model.predict(hog_features)

        print(f"{file}: {pred}")
