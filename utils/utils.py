# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import omegaconf
from typing import Iterable, List, Tuple, Iterator
from torch.nn import Module, Parameter

def flatten_dict(d, top_level_key="", sep="_"):
    """
    test = {"a": 1,
            "b": {"bb": 2},
            "c": {"cc": {"ccc":3}, "cc1": 33},
            "d": {"dd": {"dddd": {"dddd":4}}}}
    print(*[f"{k}: {v}" for k, v in utils.flatten_dict(cfg).items()], sep="\n")

    Args:
        d (_type_): _description_
        top_level_key (str, optional): _description_. Defaults to "".
        sep (str, optional): _description_. Defaults to "_".

    Returns:
        _type_: _description_
    """
    flat_d = {}
    for k, v in d.items():
        if isinstance(v, (dict, omegaconf.DictConfig)):
            flat_d.update(flatten_dict (v, top_level_key=(top_level_key + k + sep)))
        else:
            flat_d[top_level_key + k] = v
    return flat_d


def get_next_version(root_dir):
    existing_versions = []
    if not os.path.exists(root_dir):
        return 0

    for dir_name in os.listdir(root_dir):
        if dir_name.startswith("version_"):
            version_number = int(dir_name.split("_")[1])
            existing_versions.append(version_number)

    if not existing_versions:
        return 0

    return max(existing_versions) + 1


def get_weight_decay_parameters(
    named_parameters:Iterator[Parameter],
) -> Tuple[List[Parameter], List[Parameter]]:
    """Returns all parameters of the modules that should be decayed and not decayed.

    Args:
        modules:
            List of modules to get the parameters from.

    Returns:
        (params, params_no_weight_decay) tuple.
    """
    params = []
    param_names = []
    params_no_weight_decay = []
    param_names_no_weight_decay = []
    
    for n, p in named_parameters:
        if ('bias' in n) or ('bn' in n) or ('norm' in n):
            params_no_weight_decay.append(p)
            param_names_no_weight_decay.append(n)
        else:
            params.append(p)
            param_names.append(n)
    return params, params_no_weight_decay, param_names, param_names_no_weight_decay


class CosineWDSchedule(object):

    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.
    ):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd


if __name__ == '__main__':
    test = {"a": 1,
            "b": {"bb": 2},
            "c": {"cc": {"ccc":3}, "cc1": 33},
            "d": {"dd": {"dddd": {"dddd":4}}}}

    print(*[f"{k}: {v}" for k, v in flatten_dict(test).items()], sep="\n")


