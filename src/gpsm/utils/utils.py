import yaml


def load_yaml_config(path: str) -> dict:
    """Load a YAML config file as a plain dict.

    Deliberately not the `Config` class below: MotionMamba/the trainers all
    expect a plain dict (`config["n_embd"]`, `dict(config)`,
    `config.get("loss_scale", 1.0)`), and `Config` wraps values as
    attribute-access objects instead -- using it here would need changes to
    those classes too. This is the minimal, zero-risk way to stop
    hardcoding the same config dict separately in every main_*.py script.
    """
    with open(path, "r") as file:
        return yaml.safe_load(file)


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Recursively convert dictionaries to Config objects
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return cls(data)
    



def _to_jsonable(obj):
    """
    Convert common config objects into JSON-serializable structure.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    # fallback (last resort)
    return str(obj)