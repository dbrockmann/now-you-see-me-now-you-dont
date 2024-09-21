
import numpy as np
import pandas as pd

from parameters import datasets, models, model_selection
from preprocessing.data_resampling import CustomSMOTE, oversampling_strategy
from classification import test_parameters_simple
from utils.io import get_logger, load_dataset, load_parameters, save_parameters
from utils.visualization import visualize_oversampling_test


def data_oversampling(data_folder):
    """
    Optimize oversampling strategy for underrepresented classes
    """

    get_logger('data-oversampling').info('Starting data oversampling testing for the following datasets: %s; and the following models: %s.', ', '.join(datasets.keys()), ', '.join(models.keys()))

    for dataset_name in datasets:

        logger = get_logger('data-oversampling', data_folder, dataset_name)

        # load data
        train = load_dataset(data_folder, dataset_name, 'train', logger=logger)

        # set oversampling strategies
        steps = 4
        ratio = 1/1000 # oversample every class with an imbalance ratio of 1:1000 (ratio to the largest class)

        class_counts = train['y'].value_counts().sort_values(ascending=True)
        minor_classes = class_counts[class_counts / class_counts.max() < ratio]
        max_samples_class = class_counts[class_counts / class_counts.max() >= ratio].index[0]

        scores = {}

        for class_name in minor_classes.index:

            oversampling_strategies = [CustomSMOTE(steps, {class_name: step}, max_samples_class, k_neighbors=5, random_state=0) for step in range(steps)]
            test_params = {'oversampling': oversampling_strategies}
            scores[class_name] = {}

            for model_name in models:

                logger.info('Starting oversampling testing for class %s on %s.', class_name, model_name)

                # load fitted parameters
                loaded_params = load_parameters(data_folder, dataset_name, model_name, logger=logger)
                fitted_params = {'undersampling': loaded_params['undersampling'], 'feature_selection': loaded_params['feature_selection']}

                # test oversampling strategies
                test_results = test_parameters_simple(train, models[model_name], fitted_params, test_params, logger=logger, labels=[class_name], n_jobs=model_selection['n_jobs'])

                scores[class_name][model_name] = {
                    'test_mean': test_results['mean_test_f1'],
                    'test_std': test_results['std_test_f1'],
                    'test_mean_class': test_results['mean_test_f1_labels'],
                    'test_std_class': test_results['std_test_f1_labels'],
                }

            # visualize test results
            classes, counts = np.unique(train['y'], return_counts=True)
            n_samples = [oversampling_strategy(classes, counts * (1-1/model_selection['n_folds']), steps, {class_name: step}, max_samples_class)[class_name] for step in range(steps)]
            visualize_oversampling_test(scores[class_name], n_samples, data_folder, dataset_name, class_name, logger=logger)

        # select oversampling strategy by finding the step that maximizes the mean score of the models for the class
        selected_strategy = {
            class_name: np.argmax([np.mean([scores[class_name][model_name]['test_mean_class'][step] for model_name in models]) for step in range(steps)]) for class_name in minor_classes.index
        }
        new_params = {'oversampling': CustomSMOTE(steps, selected_strategy, max_samples_class, k_neighbors=5, random_state=0)}

        logger.info('Selected following oversampling strategy for the %s dataset up to class %s: %s.', dataset_name, max_samples_class, ', '.join([f'{k}: {v+1}/{steps}' for k, v in selected_strategy.items()]))

        for model_name in models:

            # load fitted parameters
            loaded_params = load_parameters(data_folder, dataset_name, model_name, logger=logger)
            fitted_params = {'undersampling': loaded_params['undersampling'], 'feature_selection': loaded_params['feature_selection']}

            # save fitted parameters
            save_parameters(fitted_params | new_params, data_folder, dataset_name, model_name, logger=logger)
