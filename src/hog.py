import cv2, os
from skimage import color, feature, io

def preprocessing(filePath):
    data_train = []
    label_train = []
    data_test = []
    label_test = []
    face = ['anger', 'happy', 'neutral', 'sad', 'suprise']  #
    for expression in face:
        directory = os.path.join(filePath, expression)  # use os.path.join for joining paths
        data_listing = os.listdir(directory)  # a list of path to each image
        order = 1
        for file in data_listing:
            # open image in the main dataset folder + subfolder, then config them
            image = io.imread(os.path.join(directory, file))
            if len(image.shape) > 2 and image.shape[2] == 3:       #only convert image has color
                image = color.rgb2gray(image)
            image = cv2.resize(image, (200, 200))

            # build hog feat.
            hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

            # add them to lists
            if order<=375:
                data_train.append(hog_features)
                label_train.append(expression)
                order+=1
            else:
                data_test.append(hog_features)
                label_test.append(expression)
                order+=1

    return data_train, label_train, data_test, label_test

