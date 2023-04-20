import os
from dynaconf import Dynaconf

parameters = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['parameters.json'],
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))