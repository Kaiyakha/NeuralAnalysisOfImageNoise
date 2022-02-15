# A script to test the network

import os
import NeuralNetwork
from GetData import *

SIZE = WIDTH, HEIGHT = 28, 28
PATH = os.path.dirname(__file__) + "/../Items"
DATA_PATH = PATH + "/Patches/Noisy_Patches/"
TRAIN_PATH = DATA_PATH + "R/"
TARGET_PATH = DATA_PATH + "strip_ids_R.csv"

X, Y = getData(TRAIN_PATH, TARGET_PATH)

network = NeuralNetwork.NeuralNetwork("dump.bin")

accuracy = network.test(X, Y)
print(f"Accuracy: {round(accuracy, 2)}%")