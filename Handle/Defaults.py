import os
from HandleConfig import getConfigDefaults

_DEFAULT_CONFIG_FILE = "config.ini"

config_defaults = getConfigDefaults(_DEFAULT_CONFIG_FILE)

ITEMS_PATH = os.path.dirname(__file__).replace("\\", "/") if config_defaults["relative"] else ""

ITEMS_PATH += config_defaults["items_path"]
if not ITEMS_PATH.endswith('/'): ITEMS_PATH += '/'
IMAGE_FILE = ITEMS_PATH + config_defaults["image_file"]
CLEAR_PATCHES = ITEMS_PATH + config_defaults["clear_patches"]
if not CLEAR_PATCHES.endswith('/'): CLEAR_PATCHES += '/'
DATASET = ITEMS_PATH + config_defaults["dataset_path"]
if not DATASET.endswith('/'): DATASET += '/'
CHANNEL = config_defaults["channel"].upper()
if CHANNEL not in ("R", "G", "B"): raise ValueError("Channel must be R, G or B")
CSV_FILENAME = lambda ch: f"strip_ids_{ch}.csv"

DEFAULT_DUMPFILE = config_defaults["dumpfile"]
DEFAULT_TRAIN_DATASET_SIZE = float(config_defaults["dataset_size"])
