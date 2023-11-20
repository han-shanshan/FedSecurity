import math
from collections import OrderedDict
from typing import Callable, List, Tuple, Dict, Any
from ..common.utils import trimmed_mean
from ..defense.defense_base import BaseDefenseMethod


class SLSGDDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.b = config.trim_param_b  # parameter of trimmed mean
        if config.alpha > 1 or config.alpha < 0:
            raise ValueError("the bound of alpha is [0, 1]")
        self.alpha = config.alpha
        self.option_type = config.option_type
        self.config = config

    def defend_before_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        extra_auxiliary_info: Any = None,
    ):
        if self.b > math.ceil(len(raw_client_grad_list) / 2) - 1 or self.b < 0:
            raise ValueError(
                "the bound of b is [0, {}])".format(
                    math.ceil(len(raw_client_grad_list) / 2) - 1
                )
            )
        if self.option_type != 1 and self.option_type != 2:
            raise Exception("Such option type does not exist!")
        if self.option_type == 2:
            raw_client_grad_list = trimmed_mean(
                raw_client_grad_list, self.b
            )  # process model list
        return raw_client_grad_list

    def defend_on_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ):
        global_model = extra_auxiliary_info
        avg_params = base_aggregation_func(args=self.config, raw_grad_list=raw_client_grad_list)
        for k in avg_params.keys():
            avg_params[k] = (1 - self.alpha) * global_model[
                k
            ] + self.alpha * avg_params[k]
        return avg_params
