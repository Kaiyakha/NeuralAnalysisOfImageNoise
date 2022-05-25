# Create artificial noise on each image
# The resulting images are in the R channel
# But other channels can also be chosen

import os, random
import csv
from typer import echo
from PIL import Image


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


def corrupt(input_path: str, output_path: str, channel: str, csv_filename: str, strip_freq: float):
	images = os.listdir(input_path)
	corrupted_image_path = output_path + channel + '/'
	csv_file_path = output_path + csv_filename
	channel = "RGB".index(channel)
	_makeDir(corrupted_image_path)

	Y = []
	for file in images:
		patch = Image.open(input_path + file)
		patch = patch.convert("RGB").split()[channel]
		strip_ids = _makeNoise(patch, strip_freq)
		Y.append(strip_ids)
		patch.save(corrupted_image_path + file, 'bmp')
		patch.close()

		ind = images.index(file)
		if ind % 100 == 0:
			percent = round(ind / len(images) * 100)
			echo(f"\rCorrupting images... {percent}%\33[0K", nl = False)

	# The ordinary number of each strip in each image gets stored into a file
	# The data is used as target data for a neural network
	with open(csv_file_path, "w", newline = "\n") as csvfile:
		fields = "Image", "Ids"
		writer = csv.DictWriter(csvfile, fieldnames = fields, delimiter = ";")
		writer.writeheader()
		for i in range(len(Y)):
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
					os.remove(path + item)
					percent = round(items.index(item) / len(items) * 100)
					echo(f"\rClearing... {percent}%\33[0K", nl = False)
			else:
				echo("Aborted!")
				exit(0)


def _makeNoise(img, strip_freq: float):
	imgMatrix = img.load()
	strip_ids = []

	for i in range(img.width):
		if random.random() < strip_freq:
			maxPix = max([imgMatrix[i, j] for j in range(img.height)])
			a = random.uniform(0, 255 / maxPix if maxPix else 0)
			b = random.randrange(0, 255 - int(a * maxPix))
			for j in range(img.height): imgMatrix[i, j] = int(a * imgMatrix[i, j] + b)
			strip_ids.append(str(i))

	strip_ids = ",".join(strip_ids)
	return strip_ids
