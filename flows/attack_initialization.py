

from parameters import datasets, models, attacks_init, features, feature_validation
from datasets import BinaryDataset
from attacking import initialize_samples
from utils.data import take_n_per_class
from utils.io import get_logger, load_dataset, load_model, save_samples


def attack_initialization(data_folder):
    """
    Perform attack methodology (no minization)
    """

    logger_flow = get_logger('attack-initialization')
    logger_flow.info('Starting sample initialization for the following datasets: %s; the following models: %s; and the following attack initializations: %s.', ', '.join(datasets.keys()), ', '.join(models.keys()), ', '.join(attacks_init.keys()))

    for dataset_name in datasets:

        # load data
        test = load_dataset(data_folder, dataset_name if not isinstance(datasets[dataset_name], BinaryDataset) else datasets[dataset_name].dataset_name, 'test', logger=logger_flow)

        # take first 1000 of each class, except benign
        test = test.loc[test['y'] != 'Benign']
        test = take_n_per_class(test, 1000)

        for model_name in models:

            # load model
            model = load_model(data_folder, dataset_name, model_name, logger=logger_flow)

            for attack_name in attacks_init:

                logger = get_logger('attack-initialization', data_folder, dataset_name, model_name, attack_name)

                # generate samples
                samples, samples_adv = initialize_samples(test, model, attacks_init[attack_name], list(features[dataset_name]), [f['min'] for f in features[dataset_name].values()], [f['max'] for f in features[dataset_name].values()], [f['discrete'] for f in features[dataset_name].values()], [f['lower'] if 'lower' in f else True for f in features[dataset_name].values()], [f['higher'] if 'higher' in f else True for f in features[dataset_name].values()],[f['base_change'] if 'base_change' in f else 0 for f in features[dataset_name].values()], feature_validation[dataset_name], logger=logger, binary_classifier=isinstance(datasets[dataset_name], BinaryDataset))

                # save samples
                save_samples((samples, samples_adv), data_folder, dataset_name, model_name, attack_name, logger=logger)
