
from parameters import datasets, models, attacks_init, attacks_min, features, feature_validation
from attacking import minimize_samples
from utils.io import get_logger, load_model, load_samples, save_samples


def attack_minimization(data_folder):
    """
    Perform attack minimization on successful adversarial examples
    """

    logger_flow = get_logger('attack-minimization')
    logger_flow.info('Starting sample minimization for the following datasets: %s; the following models: %s; the following attack initializations: %s; and the following minimization attacks: %s.', ', '.join(datasets.keys()), ', '.join(models.keys()), ', '.join(attacks_init.keys()), ', '.join(attacks_min.keys()))

    for dataset_name in datasets:

        for model_name in models:

            # load model
            model = load_model(data_folder, dataset_name, model_name, logger=logger_flow)

            for attack_init_name in attacks_init:

                # load samples
                original_samples, init_samples = load_samples(data_folder, dataset_name, model_name, attack_init_name, logger=logger_flow)

                for attack_name in attacks_min:

                    logger = get_logger('attack-minimization', data_folder, dataset_name, model_name, f'{attack_init_name}_{attack_name}')

                    # generate samples
                    samples, samples_adv = minimize_samples(original_samples, init_samples, model, attacks_min[attack_name], list(features[dataset_name]), [f['min'] for f in features[dataset_name].values()], [f['max'] for f in features[dataset_name].values()], [f['discrete'] for f in features[dataset_name].values()], [f['lower'] if 'lower' in f else True for f in features[dataset_name].values()], [f['higher'] if 'higher' in f else True for f in features[dataset_name].values()], [f['base_change'] if 'base_change' in f else 0 for f in features[dataset_name].values()], feature_validation[dataset_name], logger=logger)

                    # save samples
                    save_samples((samples, samples_adv), data_folder, dataset_name, model_name, f'{attack_init_name}_{attack_name}', logger=logger)
