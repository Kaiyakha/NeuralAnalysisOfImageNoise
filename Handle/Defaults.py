import os
from HandleConfig import getConfigDefaults

_CONFIG_FILE = "config.ini"
_IMAGERY_RANGE = "RGB"
_CLI_SPLITTER = "*"

config_defaults = getConfigDefaults(_CONFIG_FILE)

ITEMS_PATH = os.path.dirname(__file__).replace("\\", "/") if config_defaults["relative"] else ""


ITEMS_PATH += config_defaults["items_path"]
if not ITEMS_PATH.endswith('/'): ITEMS_PATH += '/'

IMAGE_FILE = ITEMS_PATH + config_defaults["image_file"]

PATCHES = ITEMS_PATH + config_defaults["patches"]
if not PATCHES.endswith('/'): PATCHES += '/'

PATCH_SIZE = config_defaults['image_resolution']
PATCH_SIZE = PATCH_SIZE.replace(config_defaults["split_char"][1:-1], _CLI_SPLITTER)

DATASET = ITEMS_PATH + config_defaults["dataset_path"]
if not DATASET.endswith('/'): DATASET += '/'

CHANNEL = config_defaults["channel"].upper()

CSV_FILENAME_TEMPLATE = config_defaults["csv_filename_template"]
CSV_FILENAME = lambda channel: f"{CSV_FILENAME_TEMPLATE}{channel}.csv"


DUMPFILE = config_defaults["dumpfile"]
TRAIN_DATASET_SIZE = float(config_defaults["train_dataset_size"])
STRIP_FREQUENCY = float(config_defaults["strip_frequency"])
OUTPUT_THRESHOLD = float(config_defaults["output_threshold"])
