_override_config_path = None
_runtime_overrides = {}

def set_override_config_path(path):
    global _override_config_path
    _override_config_path = path

def set_runtime_override(key, value):
    global _runtime_overrides
    _runtime_overrides[key] = value

def get_runtime_override(key, default=None):
    return _runtime_overrides.get(key, default)

def load_default_config():
    import yaml
    import os

    path = _override_config_path or os.path.join(os.path.dirname(__file__), "../config/default.yaml")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # apply overrides
    for key, value in _runtime_overrides.items():
        config[key] = value

    return config
