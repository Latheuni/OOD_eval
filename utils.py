# packages
import yaml


def read_config(yaml_file):
    """very sepcific yaml file

    First level categories are:
    - Main
    - Dataset
    - Network
    - Training
    """
    with open(yaml_file, "r") as file:
        file = yaml.safe_load(file)

    return file["main"], file["dataset"], file["network"], file["training"]
