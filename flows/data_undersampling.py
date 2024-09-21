
import numpy as np

from parameters import datasets, models, model_selection
from preprocessing.data_resampling import CustomRandomUnderSampler, undersampling_strategy
from classification import test_parameters_simple
from utils.io import get_logger, load_dataset, save_parameters
from utils.visualization import visualize_undersampling_test


def data_undersampling(data_folder):
    """
    Optimize random undersampling strategy
    """

    get_logger('data-undersampling').info('Starting data undersampling testing for the following datasets: %s; and the following models: %s.', ', '.join(datasets.keys()), ', '.join(models.keys()))

    for dataset_name in datasets:

        logger = get_logger('data-undersampling', data_folder, dataset_name)

        # load data
        train = load_dataset(data_folder, dataset_name, 'train', logger=logger)

        # set undersampling strategies
        steps = 10
        threshold = 0.05
        start_step = 0

        undersampling_strategies = [CustomRandomUnderSampler(steps, step, threshold, random_state=0) for step in range(start_step, steps)]
        test_params = {'undersampling': undersampling_strategies}
        scores = {}

        for model_name in models:

            logger.info('Starting undersampling testing for %s.', model_name)

            # test undersampling strategies
            test_results = test_parameters_simple(train, models[model_name], {}, test_params, logger=logger, n_jobs=model_selection['n_jobs'])

            scores[model_name] = {
                'test_mean': test_results['mean_test_f1'],
                'test_std': test_results['std_test_f1'],
            }

        # visualize test results
        classes, counts = np.unique(train['y'], return_counts=True)
        n_samples = [np.sum(list(undersampling_strategy(classes, counts * (1-1/model_selection['n_folds']), steps, step, threshold).values())) for step in range(start_step, steps)]
        visualize_undersampling_test(scores, n_samples, data_folder, dataset_name, logger=logger)

        # optimize tested parameter using results
        max_decrease = 0.01 # maximal 1 % worse than best strategy by mean, then take the maximum sample reduction

        mean_scores = [np.mean([model_scores['test_mean'][step] for model_scores in scores.values()]) for step in range(start_step, steps)]
        selected_ind = max([step for step in range(steps) if mean_scores[step] >= np.max(mean_scores) * (1-max_decrease)])
        new_params = {'undersampling': undersampling_strategies[selected_ind]}

        logger.info('Selected undersampling strategy %d/%d with %d (%.4f %%) samples for the %s dataset.', selected_ind+1, steps, n_samples[selected_ind], n_samples[selected_ind]/n_samples[0]*100, dataset_name)

        for model_name in models:

            # save fitted parameters
            save_parameters(new_params, data_folder, dataset_name, model_name, logger=logger)
