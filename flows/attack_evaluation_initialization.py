
import numpy as np

from parameters import datasets, models, attacks_init
from attacking.attack_evaluation import evaluate_attacks_init, evaluate_attacks_init_pert
from utils.io import get_logger, load_samples


def attack_evaluation_initialization(data_folder):
    """
    Evaluate adversarial examples without minimization
    """

    flow_logger = get_logger('attack-evaluation-initialization')
    flow_logger.info('Starting attack initialization evaluation for the following datasets: %s; the following models: %s; and the following attack initializations: %s.', ', '.join(datasets.keys()), ', '.join(models.keys()), ', '.join(attacks_init.keys()))

    samples, samples_adv = dict(), dict()

    for dataset_name in datasets:
        samples[dataset_name], samples_adv[dataset_name] = dict(), dict()

        for model_name in models:
            samples[dataset_name][model_name], samples_adv[dataset_name][model_name] = [], []

            for attack_name in attacks_init:

                # load samples
                orig, adv = load_samples(data_folder, dataset_name, model_name, attack_name, logger=flow_logger)

                samples[dataset_name][model_name].append(orig)
                samples_adv[dataset_name][model_name].append(adv)

    attack_name = next(iter(attacks_init)).split('_')[0]
    max_iters = [int(name.split('_')[1]) for name in attacks_init]
    max_pert = [float(f"{name.split('_')[2][0]}.{name.split('_')[2][1:]}") for name in attacks_init]

    if not np.all(np.array(max_iters) == max_iters[0]):
        evaluate_attacks_init(samples, samples_adv, list(datasets.keys()), list(models.keys()), max_iters, data_folder, attack_name, logger=flow_logger)

    if not np.all(np.array(max_pert) == max_pert[0]):
        evaluate_attacks_init_pert(samples, samples_adv, list(datasets.keys()), list(models.keys()), max_pert, data_folder, attack_name, logger=flow_logger)
