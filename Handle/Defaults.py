import os, sys
from HandleConfig import getConfigDefaults

if getattr(sys, "frozen", False): __file__ = sys.executable

_CONFIG_FILE = "config.ini"
_IMAGERY_RANGE = "RGB"
_MAX_PIX_VAL = 255
_SHIFT_DIRECTION_OPTIONS = "darker", "brighter", "any"
_CLI_SPLITTER = "*"

config_defaults = getConfigDefaults(_CONFIG_FILE)


relative_path = True if config_defaults["relative"].lower() == "true" else False

try:
    _NETWORK_PATH = os.path.dirname(__file__).replace("\\", "/") + '/' if relative_path else ""
    _NETWORK_PATH += config_defaults["network_path"]
    sys.path.append(_NETWORK_PATH)
except KeyError: pass

ITEMS_PATH = os.path.dirname(__file__).replace("\\", "/") + '/' if relative_path else ""
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
