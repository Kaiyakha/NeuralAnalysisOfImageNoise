import os
from HandleConfig import getConfigDefaults

_CONFIG_FILE = "config.ini"
_IMAGERY_RANGE = "RGB"
_MAX_PIX_VAL = 255
_CLI_SPLITTER = "*"

config_defaults = getConfigDefaults(_CONFIG_FILE)


STRIP_FREQUENCY = float(config_defaults["strip_frequency"])
MIN_SHIFT = int(config_defaults["min_deviation"])
BRIGHTER = True if config_defaults["brighter"].lower() == "true" else False
DARKER = True if config_defaults["darker"].lower() == "true" else False


relative_path = True if config_defaults["relative"].lower() == "true" else False
ITEMS_PATH = os.path.dirname(__file__).replace("\\", "/") if relative_path else ""

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
OUTPUT_THRESHOLD = float(config_defaults["output_threshold"])
