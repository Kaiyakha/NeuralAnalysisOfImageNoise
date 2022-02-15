# A script to comprise all the necessary data
# To train the neural network

import os, csv
import numpy as np
from PIL import Image

def getData(train_path: str, target_path: str):
    X = []
    for file in os.listdir(train_path):
        img = Image.open(train_path + file)
        img_matrix = np.asarray(img).reshape(img.width * img.height)
        img.close()
        X.append(img_matrix)
    X = np.array(X)

    Y = []
    with open(target_path, "r", newline = "\n") as csvfile:
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