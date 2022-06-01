# Create artificial noise on each image
# The resulting images are in the chosen channel

import os, random
import csv
from numpy import array
from typer import echo
from PIL import Image

from Defaults import _MAX_PIX_VAL, _SHIFT_DIRECTION_OPTIONS, _IMAGERY_RANGE


def crop(input_file: str, output_path: str, patch_size: tuple):
	_makeDir(output_path)

	scene = Image.open(input_file)
	width, height = patch_size
	
	for i in range(0, scene.width - width, width):
		for j in range(0, scene.height - height, height):
			patch = scene.crop((i, j, i + width, j + height))
			patch.save(output_path + f"{i}, {j}.bmp", 'bmp')
			patch.close()
		percent = round(i / (scene.width - width) * 100)
		echo(f"\rCropping... {percent}%\33[0K", nl = False)
	echo(f"\rThe image has been successfully cropped.\33[0K", nl = False)

	scene.close()


def corrupt(input_path: str, output_path: str, channel: str, csv_filename: str, config: dict):
	if channel not in _IMAGERY_RANGE: raise ValueError(f"Channel must be in range {_IMAGERY_RANGE}")
	strip_freq = float(config["strip_frequency"])
	if not 0 < strip_freq < 1: raise ValueError("Corruption probability must be in range (0, 1)")
	min_shift = int(config["min_shift"])
	if not 0 <= min_shift <= _MAX_PIX_VAL: raise ValueError(f"Minimal deviation must be in range [0, {_MAX_PIX_VAL}]")
	shift_direction = config["shift_direction"]
	if shift_direction not in _SHIFT_DIRECTION_OPTIONS: raise ValueError(f"Shift direction must be in {_SHIFT_DIRECTION_OPTIONS}")

	images = os.listdir(input_path)
	corrupted_image_path = output_path + channel + '/'
	csv_file_path = output_path + csv_filename
	channel = _IMAGERY_RANGE.index(channel)
	_makeDir(corrupted_image_path)

	Y = []
	for file in images:
		patch = Image.open(input_path + file)
		patch = patch.convert(_IMAGERY_RANGE).split()[channel]
		strip_ids = _makeNoise(patch, strip_freq, min_shift, shift_direction)
		Y.append(strip_ids)
		patch.save(corrupted_image_path + file, 'bmp')
		patch.close()

		ind = images.index(file)
		if ind % 100 == 0:
			percent = round(ind / len(images) * 100)
			echo(f"\rCorrupting images... {percent}%\33[0K", nl = False)

	# The ordinary number of each strip in each image gets stored into a file
	# The data is used as ground truth for a neural network
	with open(csv_file_path, "w", newline = "\n") as csvfile:
		fields = "Image", "Ids"
		writer = csv.DictWriter(csvfile, fieldnames = fields, delimiter = ";")
		writer.writeheader()
		for i, _ in enumerate(Y):
			writer.writerow({"Image": images[i], "Ids": Y[i]})
			if i % 100 == 0:
				percent = round(i / len(Y) * 100)
				echo (f"\rComprising data... {percent}%\33[0K", nl = False)
	echo(f"\rThe pathces have been successfully corrupted.\33[0K", nl = False)


def _makeDir(path: str):
	try: os.makedirs(path)
	except FileExistsError:
		items = os.listdir(path)

		if items:
			echo("The chosen directory already contains files. All the files in the directory will be deleted.")
			prompt = input("Type in Y/y to proceed\n").lower()

			if prompt == "y":
				for item in items:
					try: os.remove(path + item)
					except PermissionError: continue
					percent = round(items.index(item) / len(items) * 100)
					echo(f"\rClearing... {percent}%\33[0K", nl = False)
			else:
				echo("Aborted!")
				exit(0)
				

def _makeNoise(img, strip_freq: float, min_shift: int, shift_direction: str):
	imgMatrix = img.load()
	strip_ids = []

	for i in range(img.width):
		if random.random() < strip_freq:
			column = array([imgMatrix[i, j] for j in range(img.height)])
			maxPix = column.max(); minPix = column.min()

			a = random.uniform(0 if shift_direction != "brighter" else 1, \
							  ((_MAX_PIX_VAL / maxPix) if (maxPix and shift_direction != "darker") else 1))
			b = random.randint(-int(a * minPix) if shift_direction != "brighter" else 0, \
							   (_MAX_PIX_VAL - int(a * maxPix)) if shift_direction != "darker" else 0)
			if any(abs(column - (a * column + b)) < min_shift): continue

			for j in range(img.height): imgMatrix[i, j] = int(a * imgMatrix[i, j] + b)
			strip_ids.append(str(i))

	strip_ids = ",".join(strip_ids)
	return strip_ids
