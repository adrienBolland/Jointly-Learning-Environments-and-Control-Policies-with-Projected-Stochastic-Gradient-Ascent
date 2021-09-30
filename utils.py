import inspect
import importlib
import json
import numpy as np
import os
import pkgutil
import random
import torch

from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator


def import_submodules(package, recursive=True):
    """ https://stackoverflow.com/questions/3365740/how-to-import-all-submodules """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results


def get_cls(name, packages):
    for _, mod in packages.items():
        for (n, cls) in inspect.getmembers(mod, inspect.isclass):
            if n == name:
                return cls
    return None


def load_json(path):
    with open(path, "r") as json_f:
        config_dict = json.load(json_f)

    return config_dict


def get_log_path(model_name, logdir):
    log_path = None
    check_file = True
    while check_file:
        version = get_version(model_name, logdir)
        log_path = os.path.join(logdir, f"{model_name}-v{version}")

        # create a directory
        try:
            os.makedirs(log_path, exist_ok=False)
            # the directory was made successfully
            check_file = False
        except FileExistsError as e:
            # the file already exists
            version_new = get_version(model_name, logdir)

            if version == version_new:
                # the file already exists but was not created by a parallel running process using the same version
                raise e

    return log_path


def get_version(name="direct-mdp", logdir='logs/', width=3):
    """returns str repr of the version"""
    os.makedirs(logdir, exist_ok=True)
    files = list(sorted([f for f in os.listdir(logdir) if f"{name}-v" in f]))
    if len(files) < 1:
        version = '1'.rjust(width, '0')
    else:
        last_version = int(files[-1][-width:])
        version = str(last_version + 1).rjust(width, '0')
    return version


def save_config(path, config_dict):
    """drops config dictionary inside json file"""
    path = os.path.join(path, "config.json")
    return save_json(path, config_dict)


def save_json(path, dic):
    dir, file = os.path.split(path)
    if dir != '':  # current
        os.makedirs(dir, exist_ok=True)  # required if directory not created yet

    with open(path, "w") as json_f:
        json.dump(dic, json_f)


def save_error(path, exception):
    path = os.path.join(path, "error.txt")

    dir, file = os.path.split(path)
    if dir != '':  # current
        os.makedirs(dir, exist_ok=True)  # required if directory not created yet

    with open(path, "w") as file:
        file.write(str(exception))


def set_seeds(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)


def down_scale_trj(system, states, actions, rewards):
    identify = {"down_scale_state": None, "down_scale_action": None, "down_scale_reward": None, "augment_action": None}

    s = system
    for _ in range(100):
        for k in identify.keys():
            if hasattr(s, k):
                identify[k] = s
        if hasattr(s, "sys"):
            s = s.sys
        else:
            break
    states = identify["down_scale_state"].down_scale_state(states) if identify[
                                                                          "down_scale_state"] is not None else states
    actions = identify["augment_action"].augment_action(actions) if identify[
                                                                        "augment_action"] is not None else actions
    actions = identify["down_scale_action"].down_scale_action(actions) if identify[
                                                                              "down_scale_action"] is not None else actions
    rewards = identify["down_scale_reward"].down_scale_reward(rewards) if identify[
                                                                              "down_scale_reward"] is not None else rewards
    return states, actions, rewards


class CustomScale(mscale.ScaleBase):
    name = 'custom'

    def __init__(self, axis, *, min_limit=None, max_limit=150, scale_factor=5, locations=(-1000, -500, 0, 50, 100, 150), **kwargs):
        super().__init__(axis)
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.locations = locations
        self.scale_factor = scale_factor

    def get_transform(self):
        return self.CustomTransform(self.scale_factor)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(FixedLocator(self.locations))

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin if self.min_limit is None else self.min_limit, vmax if self.max_limit is None else self.max_limit

    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, scale_factor):
            mtransforms.Transform.__init__(self)
            self.scale_factor = scale_factor

        def transform_non_affine(self, a):
            return a * (1 + (np.sign(a) + 1) * self.scale_factor)

        def inverted(self):
            return CustomScale.InvertedCustomTransform(self.scale_factor)

    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, scale_factor):
            mtransforms.Transform.__init__(self)
            self.scale_factor = scale_factor

        def transform_non_affine(self, a):
            return a / (1 + (np.sign(a) + 1) * self.scale_factor)

        def inverted(self):
            return CustomScale.CustomTransform(self.scale_factor)


def list_run_dirs(logdir):
    run_dirs = [os.path.join(logdir, f) for f in os.listdir(logdir) if "run-r" in f]
    return run_dirs

