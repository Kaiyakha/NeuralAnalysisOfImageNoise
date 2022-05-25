import sys, time
import typer
from numpy import argwhere

sys.path.append("../NeuralNetwork/x64/Debug")

import NeuralNetwork as nn
from HandleConfig import getConfigInit, getConfigTrain
import Imagery
from GetData import getDataset, getVector
import Defaults


app = typer.Typer(help = "CLI for a fully connected feedforward neural network")

@app.command("init", help = "Initialise a new network")
def init_network(
	config_file: str = typer.Option(Defaults._CONFIG_FILE, "-c", "--config-file", help = "Configuration file"),
	dumpfile: str = typer.Option(Defaults.DUMPFILE, "-d", "--dumpfile", help = "A binary file to store the configuration")
):
	config_init = getConfigInit(config_file)
	network = nn.NeuralNetwork(config_init)
	network.dump(dumpfile)


@app.command("feed", help = "Get the output for a given input")
def feedforward(
	dumpfile: str = typer.Option(Defaults.DUMPFILE, "-d", "--dumpfile", help = "A binary file to load the configuration"),
	path: str = typer.Option(Defaults.DATASET + Defaults.CHANNEL + '/', "-p", "--path", help = "Path to data"),
	filename: str = typer.Argument(..., help = "Name of the file of interest"),
	threshold: float = typer.Option(Defaults.OUTPUT_THRESHOLD, "-t", "--threshold", help = "Minimum value on the output neuron to consider the answer positive")
):
	if not 0 < threshold < 1: raise ValueError("Threshold must be in range (0, 1)")

	img, img_vector = getVector(path, filename)
	network = nn.NeuralNetwork(dumpfile)
	output = network(img_vector)
	output = argwhere(output > threshold).T[0]

	print(*output)
	img.show()
	img.close()


def _getModel(dumpfile, data_path, channel, reverse_dataset, dataset_size):
	if channel not in Defaults._IMAGERY_RANGE: raise ValueError("Channel must be R, G or B")
	input_data = data_path + channel + '/'
	ground_truth = data_path + Defaults.CSV_FILENAME(channel)
	X, Y = getDataset(input_data, ground_truth)
	if reverse_dataset: X = X[::-1]; Y = Y[::-1]
	if not 0 < dataset_size <= 1: raise ValueError("dataset_size must be in the interval (0, 1]")
	dataset_size = int(X.shape[0] * dataset_size)
	network = nn.NeuralNetwork(dumpfile)
	return network, X[:dataset_size], Y[:dataset_size]


@app.command(help = "Run training")
def train(
	data_path: str = typer.Option(Defaults.DATASET, "-p", "--data-path", help = "Data for training"),
	channel: str = typer.Option(Defaults.CHANNEL, "-ch", "--channel", help = f"A channel from the {Defaults._IMAGERY_RANGE} range"),
	reverse_dataset: bool = typer.Option(False, "-r", "--reverse-dataset", help = "Invert the order of the elements in the dataset"),
	dataset_size: float = typer.Option(Defaults.TRAIN_DATASET_SIZE, "-s", "--dataset-size", help = "Multiply the size of the dataset by the value"),
	config_file: str = typer.Option(Defaults._CONFIG_FILE, "-c", "--config-file", help = "Configuration file"),
	dumpfile: str = typer.Option(Defaults.DUMPFILE, "-d", "--dumpfile", help = "A binary file to store and load the configuration"),
	dump: bool = typer.Option(True, "--no-dump", help = "Do not dump the configuration", show_default = False)
):
	typer.echo("Initialising...", nl = False)
	config_train = getConfigTrain(config_file)
	network, X, Y = _getModel(dumpfile, data_path, channel, reverse_dataset, dataset_size)
	
	typer.echo("\rTraining...\33[0K")
	train_time = time.time()
	network.train(X, Y, config_train)
	train_time = round(time.time() - train_time)
	m, s = divmod(train_time, 60)
	h, m = divmod(m, 60)
	print(f"\nSession time: {str(h).zfill(2)}:{str(m).zfill(2)}:{str(s).zfill(2)}")

	if dump: network.dump(dumpfile)


@app.command(help = "Run test")
def test(
	data_path: str = typer.Option(Defaults.DATASET, "-p", "--data-path", help = "Data for testing"),
	channel: str = typer.Option(Defaults.CHANNEL, "-ch", "--channel", help = f"A channel from the {Defaults._IMAGERY_RANGE} range"),
	reverse_dataset: bool = typer.Option(True, "-nr", "--no-reverse-dataset", show_default = False, help = "Do not invert the order of the elements in the dataset"),
	dataset_size: float = typer.Option(round(1 - Defaults.TRAIN_DATASET_SIZE, 2), "-s", "--dataset-size", help = "Multiply the size of the dataset by the value"),
	dumpfile: str = typer.Option(Defaults.DUMPFILE, "-d", "--dumpfile", help = "A binary file to load the configuration")
):
	typer.echo("Initialising...", nl = False)
	network, X, Y = _getModel(dumpfile, data_path, channel, reverse_dataset, dataset_size)
	typer.echo("\rTesting...\33[0K")
	accuracy = network.test(X, Y)
	typer.echo(f"Accuracy: {round(accuracy, 2)}%")


@app.command(help = "Crop an image into patches")
def crop(
	input_file: str = typer.Option(Defaults.IMAGE_FILE, "-i", "--input-path", help = "Path to an image to crop"),
	output_path: str = typer.Option(Defaults.PATCHES, "-o", "--output-path", help = "Output path"),
	patch_size: str = typer.Option(Defaults.PATCH_SIZE, "-s", "--patch-size", help = f"Size of each patch, values must be separated with {Defaults._CLI_SPLITTER}")
):
	patch_size = tuple([int(value) for value in patch_size.split(Defaults._CLI_SPLITTER)])
	Imagery.crop(input_file, output_path, patch_size)


@app.command(help = "Corrupt the images with vertical strips")
def corrupt(
	input_path: str = typer.Option(Defaults.PATCHES, "-i", "--input-path", help = "Path to images to corrupt"),
	output_path: str = typer.Option(Defaults.DATASET, "-o", "--output-path", help = "Path to store the output to"),
	channel: str = typer.Argument(..., help = "A channel from the RGB range to provide the result"),
	csv_filename_template: str = typer.Option(Defaults.CSV_FILENAME_TEMPLATE, "-f", "--csv-filename", help = "CSV filename template"),
	strip_freq: float = typer.Option(Defaults.STRIP_FREQUENCY, "-sf", "--strip-freq", help = "Corruption probability for a single strip")
):
	if channel not in Defaults._IMAGERY_RANGE: raise ValueError("Channel must be R, G or B")
	if not 0 < strip_freq < 1: raise ValueError("Corruption probability must be in range (0, 1)")
	csv_filename = Defaults.CSV_FILENAME(channel)
	Imagery.corrupt(input_path, output_path, channel, csv_filename, strip_freq)


if __name__ == "__main__":
	app()