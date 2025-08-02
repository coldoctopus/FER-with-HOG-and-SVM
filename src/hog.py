import cv2, os
from skimage import color, feature, io

def preprocessing_split(filePath, train_ratio):
    data_train = []
    label_train = []
    data_test = []
    label_test = []
    face = ['anger', 'happy', 'neutral', 'sad', 'suprise']  # list of expressions

    for expression in face:
        directory = os.path.join(filePath, expression)  # use os.path.join for joining paths
        data_listing = os.listdir(directory)  # a list of path to each image
        total_images = len(data_listing)
        train_count = int(train_ratio * total_images)

        # catch the error if not having enough images for train or test
        if train_count == 0:
            raise ValueError(f"Train ratio too low — no images left for training in class '{expression}'")
        if total_images - train_count == 0:
            raise ValueError(f"Train ratio too high — no images left for testing in class '{expression}'")
        
        order = 1  # to count the number of images processed
        for file in data_listing:
            # open image in the main dataset folder + subfolder, then config them
            image = io.imread(os.path.join(directory, file))
            if len(image.shape) > 2 and image.shape[2] == 3:       #only convert image has color
                image = color.rgb2gray(image)
            image = cv2.resize(image, (200, 200))

            # build hog features
            hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

            # add them to lists
            if order<=train_count:
                data_train.append(hog_features)
                label_train.append(expression)
                order+=1
            else:
                data_test.append(hog_features)
                label_test.append(expression)
                order+=1

    return data_train, label_train, data_test, label_test

def preprocessing_full(filePath):
    data = []
    label = []
    face = ['anger', 'happy', 'neutral', 'sad', 'suprise']  #
    for expression in face:
        directory = os.path.join(filePath, expression)  # use os.path.join for joining paths
        data_listing = os.listdir(directory)  # a list of path to each image
        for file in data_listing:
            # open image in the main dataset folder + subfolder, then config them
            image = io.imread(os.path.join(directory, file))
            if len(image.shape) > 2 and image.shape[2] == 3:       #only convert image has color
                image = color.rgb2gray(image)
            image = cv2.resize(image, (200, 200))

            # build hog feat.
            hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

            data.append(hog_features)
            label.append(expression)

    return data, label
