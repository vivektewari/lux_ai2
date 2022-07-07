
from monobeast_lux_ai import main
from types import SimpleNamespace
import yaml

with open('config.yaml', 'r') as f:
    config = SimpleNamespace(**yaml.safe_load(f))
flags = config
main(flags)