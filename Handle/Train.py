# A script to train the network
# Once the training is done, the configuration of the network
# gets stored into a .pkl file so that it can be uploaded
# to continue the training on the next run of the script

import os, cursor, time
from HandleConfig import getConfigInit, getConfigTrain
from GetData import getData
from NeuralNetwork import *

PATH = os.path.dirname(__file__) + "/../Items"
DATA_PATH = PATH + "/Patches/Noisy_Patches/"
TRAIN_PATH = DATA_PATH + "R/"
TARGET_PATH = DATA_PATH + "strip_ids_R.csv"
CONFIG_FILE = "config.ini"

cursor.hide()
print("Initialising...", end = '\r')

config_init, config_train = getConfigInit(CONFIG_FILE), getConfigTrain(CONFIG_FILE)
X, Y = getData(TRAIN_PATH, TARGET_PATH)

network = NeuralNetwork(config_init)
    
print("Training...\t")
train_time = time.time()
network.train(X[:-1000], Y[:-1000], config_train)
train_time = round(time.time() - train_time)
m, s = divmod(train_time, 60)
h, m = divmod(m, 60)
print(f"\nTime taken: {str(h).zfill(2)}:{str(m).zfill(2)}:{str(s).zfill(2)}")
print("\nTesting...")
accuracy = network.test(X[-1000:], Y[-1000:])
print(f"Accuracy: {round(accuracy, 2)}%")