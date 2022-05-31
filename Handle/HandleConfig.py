from configparser import ConfigParser
from psutil import cpu_count


def _getConfig(config_file: str) -> ConfigParser:
	...


def getConfigInit(config_file: str) -> dict:
	config = _getConfig(config_file)    

	split_char = config["meta"]["split_char"][1:-1]

	config_init = dict(config.items("image_resolution") + config.items("network_configuration"))
	for key in config_init: config_init[key] = config_init[key].split(split_char)
	
	WIDTH, HEIGHT = [abs(int(val)) for val in config_init["image_resolution"]]
	shape_modifiers = [abs(float(val)) for val in config_init["shape_modifiers"]]
	del config_init["shape_modifiers"]

	config_init["shape"] = [WIDTH * HEIGHT]
	for modifier in shape_modifiers:
		layer_width = round(modifier * WIDTH)
		config_init["shape"].append(layer_width)
	config_init["shape"].append(WIDTH)

	config_init["activation_function_parameters"] = [float(val) for val in config_init["activation_function_parameters"]]

	return config_init


def getConfigTrain(config_file: str) -> dict:
	config = _getConfig(config_file)

	train_parameters = dict(config.items("train_parameters"))
	for key in train_parameters: train_parameters[key] = float(train_parameters[key])

	dynamic_rate = dict(config.items("dynamic_rate"))
	for key in "rate_delta", "accuracy_stuck_limit", "accuracy_stuck_limit_delta":
		dynamic_rate[key] = float(dynamic_rate[key])

	parallel_training = dict(config.items("parallel_training"))
	threads = parallel_training["threads"]
	parallel_training["threads"] = \
		cpu_count(logical = True) if threads.lower() == "auto" \
		else cpu_count(logical = False) if threads.lower() == "cores" \
		else int(threads)

	config_train = dict(train_parameters, **dynamic_rate, **parallel_training)
	for key in "parallel_training", "dynamic_rate":
		config_train[key] = True if config_train[key].lower() in ("true", "enable") else False

	return config_train


def getConfigDefaults(config_file: str) -> dict:
	config = _getConfig(config_file)

	config_defaults = dict(config.items("meta") \
						   + config.items("image_resolution") \
						   + config.items("corruption") \
						   + config.items("default_path") \
						   + config.items("defaults") \
						   )

	return config_defaults


def _getConfig(config_file: str) -> ConfigParser:
	config = ConfigParser()
	config.read(config_file)
	return config