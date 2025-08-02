import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

def transform_data(data, labels):
    #Generate input array X
    X = np.array(data)

    #Generate output array y
    lbe = LabelEncoder()
    lbe.fit(labels)
    y = lbe.transform(labels)

    return X, y

def show_plot(title,data):
    face = ['anger', 'happy', 'neutral', 'sad', 'suprise']

    # Create a bar chart
    bars = plt.bar(face, data, color=['red', 'lightgreen', 'yellow', 'gray', 'plum'])

    # Add labels and title
    plt.xlabel('Expression')
    plt.ylabel('Percentage (%)')
    plt.title(title)

    for bar in bars:
        val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, val, round(val, 2), ha='center', va='bottom')
    # Show the bar chart
    plt.show()