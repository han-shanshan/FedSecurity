import math
from collections import OrderedDict
from typing import Callable, List, Tuple, Dict, Any

from ..common.bucket import Bucket
from ..common.utils import compute_geometric_median
from ...security.defense.defense_base import BaseDefenseMethod


class GeometricMedianDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.byzantine_client_num = config.byzantine_client_num
        self.client_num_per_round = config.client_num_per_round
        # 2(1 + ε )q ≤ batch_num ≤ client_num_per_round
        # trade-off between accuracy & robustness:
        #       larger batch_num --> more Byzantine robustness, larger estimation error.
        self.batch_num = config.batch_num
        if self.byzantine_client_num == 0:
            self.batch_num = 1
        self.batch_size = math.ceil(self.client_num_per_round / self.batch_num)

    def defend_on_aggregation(
            self,
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            base_aggregation_func: Callable = None,
            extra_auxiliary_info: Any = None,
    ):
        batch_grad_list = Bucket.bucketization(raw_client_grad_list, self.batch_size)
        (num0, avg_params) = batch_grad_list[0]
        alphas = {alpha for (alpha, params) in batch_grad_list}
        alphas = {alpha / sum(alphas, 0.0) for alpha in alphas}
        for k in avg_params.keys():
            batch_grads = [params[k] for (alpha, params) in batch_grad_list]
            avg_params[k] = compute_geometric_median(alphas, batch_grads)
        return avg_params


