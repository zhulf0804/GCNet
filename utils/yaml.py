import yaml


def decode_config(filepath):
    with open(filepath, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    configs_dict = dict()
    for k, v in data.items():
        for sub_k, sub_v in v.items():
            configs_dict[sub_k] = sub_v
    return configs_dict
