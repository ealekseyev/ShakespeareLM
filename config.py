import yaml
import os

_cfg = None

def get_config():
    global _cfg
    if _cfg is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, "r") as f:
            _cfg = yaml.safe_load(f)
    return _cfg
