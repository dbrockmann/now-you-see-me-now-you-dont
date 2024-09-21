

from parameters import datasets, models
from classification import evaluate_model
from utils.io import get_logger, load_dataset, load_model


def model_evaluation(data_folder):
    """
    Evaluate models on the test set
    """

    base_logger = get_logger('model-evaluation')
    base_logger.info('Starting model evaluation for the following datasets: %s; and the following models: %s.', ', '.join(datasets.keys()), ', '.join(models.keys()))

    for dataset_name in datasets:

        # load data
        train = load_dataset(data_folder, dataset_name, 'train', logger=base_logger)
        test = load_dataset(data_folder, dataset_name, 'test', logger=base_logger)
        
        for model_name in models:

            logger = get_logger('model-evaluation', data_folder, dataset_name, model_name)

            # load model
            fitted_model = load_model(data_folder, dataset_name, model_name, logger=logger)

            # evaluate model
            evaluate_model(train, test, fitted_model, data_folder, dataset_name, model_name, logger=logger)
