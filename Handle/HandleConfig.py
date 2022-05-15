from configparser import ConfigParser


def getConfig(config_file) -> tuple:
    config = ConfigParser()
    config.read(config_file)

    split_char = config["meta"]["split_char"][1:-1]

    config_init = dict(config.items("network_configuration"))
    for key in config_init: config_init[key] = config_init[key].split(split_char)
    
    WIDTH, HEIGHT = config_init["image_resolution"]
    for i in range(len(config_init["shape_modifiers"])):
        modifier = round(float(config_init["shape_modifiers"][i]) * int(WIDTH))
        config_init["shape_modifiers"][i] = str(modifier)
    config_init["shape_modifiers"].insert(0, str(int(WIDTH) * int(HEIGHT)))
    config_init["shape_modifiers"].append(WIDTH)
    
    config_train = dict(config.items("train_parameters") + config.items("dynamic_rate") + config.items("parallel_training"))
    ENABLE = "true", "True", "enable", "Enable"
    if config_train["dynamic_rate"] not in ENABLE:
        config_train["dynamic_rate"] = False
    if (config_train["parallel_training"] not in ENABLE) or (int(config_train["threads"]) < 2):
        config_train["parallel_training"] = False

    return config_init, config_train