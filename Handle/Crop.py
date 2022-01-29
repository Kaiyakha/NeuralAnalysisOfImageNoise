
# A script to cut the image to pieces

from PIL import Image
import os

SIZE = WIDTH, HEIGHT = 28, 28
PATH = os.path.dirname(__file__) + '/../Items/'
PATCHES_PATH = PATH + "Patches/Clear_Patches/"
FILE = "vd1.bmp"
try: os.makedirs(PATCHES_PATH)
except FileExistsError: pass

scene = Image.open(PATH + FILE)

BAR_CAPACITY = 300
BARS = (scene.width - WIDTH) // BAR_CAPACITY
if BARS % BAR_CAPACITY == 0: BARS -= 1

print(f"\rCropping |{'-' * BARS}|\0", end = '')
for i in range(0, scene.width - WIDTH, WIDTH):
    if i % BAR_CAPACITY < WIDTH:
        loaded = i // BAR_CAPACITY; left = BARS - loaded
        print(f"\rCropping |{'█' * loaded}{'-' * left}|\0", end = '')
    for j in range(0, scene.height - HEIGHT, HEIGHT):
        patch = scene.crop((i, j, i + WIDTH, j + HEIGHT))
        patch.save(PATCHES_PATH + f"{i}, {j}.bmp", 'bmp')
        patch.close()

scene.close()