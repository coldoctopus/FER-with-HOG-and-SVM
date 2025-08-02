import cv2
import joblib
import numpy as np
from skimage import color, feature
from datetime import datetime

# Load the trained model
ker = 'poly'  # kernel type, change to 'linear', 'rbf', 'sigmoid' or 'poly' for different kernels
deg = 3
gam = 0.1
c = 75
coef = 1
date = '02082025'   #ddmmyyyy: date of the model
time = '104313'   #hhmmss: time of the model

model = joblib.load(f'../models/kernel-{ker}_deg-{deg}_gamma-{gam}_C-{c}_coef0-{coef}_{date}_{time}.pkl')

# Open webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale if needed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize to the size used during training
    resized = cv2.resize(gray, (200, 200))

    # Extract HOG features
    hog_features = feature.hog(resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

    # Reshape for prediction
    hog_features = hog_features.reshape(1, -1)

    # Predict
    prediction = model.predict(hog_features)[0]

    # Draw the prediction on the frame
    cv2.putText(frame, f'Prediction: {prediction}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Live Expression Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
