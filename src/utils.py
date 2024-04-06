""" General utilities for the project."""

import pathlib
import pickle
import configparser


def get_proj_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


def save_value(value, fname: pathlib.Path):
    fname.parent.mkdir(parents=True, exist_ok=True)
    with open(fname, "wb") as f:
        pickle.dump(value, f)


def load_value(fname: pathlib.Path):
    with open(fname, "rb") as f:
        value = pickle.load(f)
    return value


def get_config(config_rel_path: pathlib.Path, interpolation=None):

    proj_root = get_proj_root()

    config = configparser.ConfigParser(interpolation=interpolation)
    config.read(proj_root.joinpath(config_rel_path))

    return config


def get_full_path(rel_path: pathlib.Path):

    proj_root = get_proj_root()
    full_path = proj_root.joinpath(rel_path)

    return full_path
