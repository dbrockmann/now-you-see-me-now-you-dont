
import numpy as np
from feature_engine.selection import DropCorrelatedFeatures
from sklearn.base import clone

from parameters import datasets, models, model_selection
from classification import test_parameters_simple
from utils.io import get_logger, load_dataset, load_parameters, save_parameters
from utils.visualization import visualize_feature_selection_test


def data_feature_selection(data_folder):
    """
    Optimize feature selection approach
    """

    get_logger('data-feature-selection').info('Starting data feature selection testing for the following datasets: %s; and the following models: %s.', ', '.join(datasets.keys()), ', '.join(models.keys()))

    for dataset_name in datasets:

        logger = get_logger('data-feature-selection', data_folder, dataset_name)

        # load data
        train = load_dataset(data_folder, dataset_name, 'train', logger=logger)

        # set feature selection methods, first should be 1.0
        thresholds = np.linspace(1.0, 0.0, num=10, endpoint=False)

        selection_methods = [DropCorrelatedFeatures(threshold=threshold) for threshold in thresholds]
        test_params = {'feature_selection': selection_methods}
        scores = {}

        for model_name in models:

            logger.info('Starting feature selection testing for %s.', model_name)

            # load fitted parameters
            fitted_params = {'undersampling': load_parameters(data_folder, dataset_name, model_name, logger=logger)['undersampling']}

            # test feature selection methods
            test_results = test_parameters_simple(train, models[model_name], fitted_params, test_params, logger=logger, n_jobs=model_selection['n_jobs'])

            scores[model_name] = {
                'train_mean': test_results['mean_train_f1'],
                'train_std': test_results['std_train_f1'],
                'test_mean': test_results['mean_test_f1'],
                'test_std': test_results['std_test_f1'],
            }

        # visualize test results
        fitted_methods = [clone(selection_method).fit(train.drop(columns='y')) for selection_method in selection_methods]
        n_features = [method.n_features_in_ - len(method.features_to_drop_) for method in fitted_methods]
        visualize_feature_selection_test(scores, thresholds, n_features, data_folder, dataset_name, logger=logger)

        # optimize tested parameter using results
        max_decrease = 0.01 # maximal 1 % worse than best strategy by mean, then take the maximum feature reduction

        mean_scores = [np.mean([model_scores['test_mean'][step] for model_scores in scores.values()]) for step in range(len(thresholds))]
        selected_ind = max([step for step in range(len(thresholds)) if mean_scores[step] >= np.max(mean_scores) * (1-max_decrease)])
        new_params = {'feature_selection': selection_methods[selected_ind]}

        logger.info('Selected feature selection method %f with %d/%d (%.4f %%) features for the %s dataset.', thresholds[selected_ind], n_features[selected_ind], n_features[0], n_features[selected_ind]/n_features[0]*100, dataset_name)

        for model_name in models:

            # load fitted parameters
            fitted_params = {'undersampling': load_parameters(data_folder, dataset_name, model_name, logger=logger)['undersampling']}

            # save fitted parameters
            save_parameters(fitted_params | new_params, data_folder, dataset_name, model_name, logger=logger)
