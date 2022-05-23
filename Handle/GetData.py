# A script to comprise all the necessary data
# To train the neural network

import os, csv
import numpy as np
from PIL import Image


def getVector(path: str, filename: str):
    img = Image.open(path + filename)
    img_vector = np.asarray(img).reshape(img.width * img.height)
    return img, img_vector


def getDataset(input_path: str, ground_truth_path: str):
    X = []
    for file in os.listdir(input_path):
        img, img_vector = getVector(input_path, file)
        img.close()
        X.append(img_vector)
    X = np.array(X)

    Y = []
    with open(ground_truth_path, "r", newline = "\n") as csvfile:
        reader = csv.DictReader(csvfile, delimiter = ";")
        for row in reader:
            y = row["Ids"]
            if len(y): y = np.array(y.split(","), dtype = "int8")
            else: y = np.array([])
            Y.append(y)

    for i in range(len(Y)):
        y = np.zeros(img.width)  
        for j in Y[i]:
            y[j] = 1
        Y[i] = y
    Y = np.array(Y)

    return X, Y