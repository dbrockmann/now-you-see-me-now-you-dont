
import sys
import logging
import traceback

from flows import data_cleaning, data_load_binary, data_undersampling, data_feature_selection, data_oversampling, model_parameter_fitting, model_retraining, model_evaluation, attack_initialization, attack_minimization, attack_evaluation_initialization, attack_evaluation_minimization, plot_additionals
from utils.io import configure_logging


# folder to store artifacts (e.g., processed data, models)
data_folder = 'data/'


flows = {
    'data-cleaning': data_cleaning,
    'data-load-binary': data_load_binary,
    'data-undersampling': data_undersampling,
    'data-feature-selection': data_feature_selection,
    'data-oversampling': data_oversampling,

    'model-parameter-fitting': model_parameter_fitting,
    'model-retraining': model_retraining,
    'model-evaluation': model_evaluation,

    'attack-initialization': attack_initialization,
    'attack-minimization': attack_minimization,
    'attack-evaluation-initialization': attack_evaluation_initialization,
    'attack-evaluation-minimization': attack_evaluation_minimization,
    
    'plot-additionals': plot_additionals
}

def main(flow_names):
    """
    Starts each flow given by name, if None is given run all flows.

    Args:
        flow_names: List of flow names.
    """

    if flow_names is None:
        for flow in flows.values():
            flow(data_folder=data_folder)

    else:
        for flow_name in flow_names:
            if flow_name in flows:
                flows[flow_name](data_folder=data_folder)
            else:
                logging.error('Invalid arguments. Use no argument or at least one of the following: %s.', ', '.join(flows.keys()))

if __name__ == '__main__':
    configure_logging()
    try:
        main(None if len(sys.argv) == 1 else sys.argv[1:])
    except Exception as e:
        logging.error(traceback.format_exc())
