import yaml
from pathlib import Path


def load_config(path: str):
    """
    Load a YAML configuration file.
    """
    config_path = Path(path)
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    return config
