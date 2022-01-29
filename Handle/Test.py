
# A script to test the network

import os, dill
from GetData import *

SIZE = WIDTH, HEIGHT = 28, 28
PATH = os.path.dirname(__file__) + "/../Items"
DATA_PATH = PATH + "/Patches/Noisy_Patches/"
TRAIN_PATH = DATA_PATH + "R/"
TARGET_PATH = DATA_PATH + "strip_ids_R.csv"

X, Y = getData(TRAIN_PATH, TARGET_PATH)

with open(PATH + "/trained_nn.pkl", "rb") as pklfile:
    nn = dill.load(pklfile)

nn.test(X, Y)