# A script to train the network
# Once the training is done, the configuration of the network
# gets stored into a .pkl file so that it can be uploaded
# to continue the training on the next run of the script

import os, cursor, dill, time
from GetData import *
from NeuralNetwork import *

SIZE = WIDTH, HEIGHT = 28, 28
PATH = os.path.dirname(__file__) + "/../Items"
DATA_PATH = PATH + "/Patches/Noisy_Patches/"
TRAIN_PATH = DATA_PATH + "R/"
TARGET_PATH = DATA_PATH + "strip_ids_R.csv"

X, Y = getData(TRAIN_PATH, TARGET_PATH)

try:
    with open(PATH + "/trained_nn.pkl", "rb") as pklfile:
        nn = dill.load(pklfile)
except FileNotFoundError:
    nn = NeuralNetwork((WIDTH * HEIGHT, WIDTH * 3, int(WIDTH * 1.8),  WIDTH))
    
cursor.hide()
print("\nTraining...")
train_time = time.time()
nn.train(X[:-1000], Y[:-1000], 1e-4, int(1e7), int(1e3))
train_time = round(time.time() - train_time)
print("\nTesting...")
nn.test(X[-1000:], Y[-1000:])
m, s = divmod(train_time, 60)
h, m = divmod(m, 60)
print(f"\nTime taken: {str(h).zfill(2)}:{str(m).zfill(2)}:{str(s).zfill(2)}")

# with open(PATH + "/trained_nn.pkl", "wb") as pklfile:
#     dill.dump(nn, pklfile, dill.HIGHEST_PROTOCOL)