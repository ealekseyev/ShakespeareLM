import yaml
import os


def get_config(version=None):
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "config.yaml")) as f:
        root_cfg = yaml.safe_load(f)
    v = version or root_cfg["active_version"]
    version_yaml = os.path.join(root, "versions", v, "version.yaml")
    with open(version_yaml) as f:
        cfg = yaml.safe_load(f)
    # Resolve paths relative to repo root
    for key in ("token_dir", "checkpoint_dir", "model_file"):
        cfg[key] = os.path.join(root, cfg[key])
    if not os.path.isabs(cfg.get("corpus_file", "")):
        cfg["corpus_file"] = os.path.join(root, cfg["corpus_file"])
    cfg["version"] = v
    cfg["root"] = root
    return cfg
