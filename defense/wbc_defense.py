from collections import OrderedDict
import torch
from typing import Callable, List, Tuple, Dict, Any
import numpy as np
import logging
from .defense_base import BaseDefenseMethod
from ..common import utils


class WbcDefense(BaseDefenseMethod):
    def __init__(self, args):
        self.args = args
        self.client_idx = args.client_idx
        self.batch_idx = args.batch_idx
        self.old_gradient = {}

    def run(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ) -> Dict:
        num_client = len(raw_client_grad_list)
        vec_local_w = [
            (
                raw_client_grad_list[i][0],
                utils.vectorize_weight(raw_client_grad_list[i][1]),
            )
            for i in range(0, num_client)
        ]

        # extra auxiliary information: model parameters at current round -> dict
        models_param = extra_auxiliary_info
        model_param = models_param[self.client_idx][1]

        new_model_param = {}
        if self.batch_idx != 0:
            for (k, v) in model_param.items():
                if "weight" in k:
                    grad_tensor = (
                        raw_client_grad_list[self.client_idx][1][k].cpu().numpy()
                    )
                    # for testing, simply pre-defin old gradient
                    self.old_gradient[k] = grad_tensor * 0.2
                    grad_diff = grad_tensor - self.old_gradient[k]
                    pert_strength = 1
                    pertubation = np.random.laplace(
                        0, pert_strength, size=grad_tensor.shape
                    ).astype(np.float32)
                    pertubation = np.where(
                        abs(grad_diff) > abs(pertubation), 0, pertubation
                    )
                    learning_rate = 0.1
                    new_model_param[k] = torch.from_numpy(
                        model_param[k].cpu().numpy() + pertubation * learning_rate
                    )
                else:
                    new_model_param[k] = model_param[k]
        for (k, v) in model_param.items():
            if "weight" in k:
                self.old_gradient[k] = (
                    raw_client_grad_list[self.client_idx][1][k].cpu().numpy()
                )

        param_list = []
        for i in range(0, num_client):
            if i != self.client_idx or self.batch_idx == 0:
                param_list.append(models_param[i])
            else:
                param_list.append((models_param[self.client_idx][0], new_model_param))
                logging.info(f"New. param: {param_list[i]}")

        return base_aggregation_func(self.args, param_list)  # avg_params
