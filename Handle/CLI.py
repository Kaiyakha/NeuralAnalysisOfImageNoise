import sys, time
import typer

sys.path.append("../NeuralNetwork/x64/Debug")

import NeuralNetwork as nn
from HandleConfig import getConfigInit, getConfigTrain
from GetData import getData
import Defaults


app = typer.Typer()

@app.command("init", help = "Initialise a new network")
def init_network(
    config_file: str = typer.Option(Defaults._DEFAULT_CONFIG_FILE, "-c", "--config-file", help = "Configuration file"),
    dumpfile: str = typer.Option(Defaults.DEFAULT_DUMPFILE, "-d", "--dumpfile", help = "A binary file to store the configuration")
):
    config_init = getConfigInit(config_file)
    network = nn.NeuralNetwork(config_init)
    network.dump(dumpfile)


def _getModel(dumpfile, data_path, channel, reverse_dataset, dataset_size):
    input_data = data_path + channel + '/'
    ground_truth = data_path + Defaults.CSV_FILENAME(channel)
    X, Y = getData(input_data, ground_truth)
    if reverse_dataset: X = X[::-1]; Y = Y[::-1]
    if not 0 < dataset_size <= 1: raise ValueError("dataset_size must be in the interval (0, 1]")
    dataset_size = int(X.shape[0] * dataset_size)
    network = nn.NeuralNetwork(dumpfile)
    return network, X[:dataset_size], Y[:dataset_size]


@app.command(help = "Run training")
def train(
    data_path: str = typer.Option(Defaults.DATASET, "-p", "--data-path", help = "Data for training"),
    channel: str = typer.Option(Defaults.CHANNEL, "-ch", "--channel", help = "A channel from the RGB range"),
    reverse_dataset: bool = typer.Option(False, "-r", "--reverse-dataset", help = "Invert the order of the elements in the dataset"),
    dataset_size: float = typer.Option(Defaults.DEFAULT_TRAIN_DATASET_SIZE, "-s", "--dataset-size", help = "Multiply the size of the dataset by the value"),
    config_file: str = typer.Option(Defaults._DEFAULT_CONFIG_FILE, "-c", "--config-file", help = "Configuration file"),
    dumpfile: str = typer.Option(Defaults.DEFAULT_DUMPFILE, "-d", "--dumpfile", help = "A binary file to store and load the configuration"),
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
    print(f"\nTime taken: {str(h).zfill(2)}:{str(m).zfill(2)}:{str(s).zfill(2)}")

    if dump: network.dump(dumpfile)


@app.command(help = "Run test")
def test(
    data_path: str = typer.Option(Defaults.DATASET, "-p", "--data-path", help = "Data for testing"),
    channel: str = typer.Option(Defaults.CHANNEL, "-ch", "--channel", help = "A channel from the RGB range"),
    reverse_dataset: bool = typer.Option(True, "-nr", "--no-reverse-dataset", show_default = False, help = "Do not invert the order of the elements in the dataset"),
    dataset_size: float = typer.Option(round(1 - Defaults.DEFAULT_TRAIN_DATASET_SIZE, 2), "-s", "--dataset-size", help = "Multiply the size of the dataset by the value"),
    dumpfile: str = typer.Option(Defaults.DEFAULT_DUMPFILE, "-d", "--dumpfile", help = "A binary file to load the configuration")
):
    typer.echo("Initialising...", nl = False)
    network, X, Y = _getModel(dumpfile, data_path, channel, reverse_dataset, dataset_size)
    typer.echo("\rTesting...\33[0K")
    accuracy = network.test(X, Y)
    typer.echo(f"Accuracy: {round(accuracy, 2)}%")


if __name__ == "__main__":
    app()