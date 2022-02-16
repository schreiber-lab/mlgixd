# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from ..config import Config, SimMetricsConfig

from .calc_metrics import get_sim_metrics
from .precision_recall_curve import get_precision_recall_curve, get_max_accuracy_metrics


def get_sim_metrics_from_config(model, sim, config: Config) -> dict:

    res_dict = {
        'sim_metrics': []
    }

    sim_params = _sim_metrics_config_to_list(config.sim_metrics)

    for params in sim_params:

        sim_metrics = get_sim_metrics(model, sim, disable_tqdm=config.general.disable_tqdm, **params)
        recalls, precisions, accuracies = get_precision_recall_curve(sim_metrics)
        recall, precision, accuracy = get_max_accuracy_metrics(recalls, precisions, accuracies)

        res_dict['sim_metrics'].append({
            'params': params,
            'metrics': sim_metrics.state_dict(),
            'recalls': recalls,
            'precisions': precisions,
            'accuracies': accuracies,
            'best_metrics': {
                'recall': recall,
                'precision': precision,
                'accuracy': accuracy,
            }
        })

    return res_dict


def _sim_metrics_config_to_list(sim_metrics_config: SimMetricsConfig):

    d = {k: [v] if not isinstance(v, (tuple, list)) else v
         for k, v in sim_metrics_config.asdict().items()}

    lengths = set(len(v) for v in d.values())
    max_length = max(lengths)

    if lengths in ({1}, {max_length}, {1, max_length}):
        conf_dicts = [{k: v[i] if len(v) > 1 else v[0] for k, v in d.items()} for i in range(max(lengths))]
    else:
        raise ValueError('SimMetricsConfig values are not consistent.')
    return conf_dicts
