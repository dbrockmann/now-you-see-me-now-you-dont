

import pandas as pd
from utils.data import split_data
from parameters import datasets, models
from classification import retrain_model
from utils.io import get_logger, load_dataset, load_parameters, save_model
from utils.visualization import visualize_retraining

def model_retraining(data_folder):
    """
    Retrain models with fitted parameters on the training data
    """

    get_logger('model-retraining').info('Starting model retraining for the following datasets: %s; and the following models: %s.', ', '.join(datasets.keys()), ', '.join(models.keys()))

    for dataset_name in datasets:

        for model_name in models:

            logger = get_logger('model-retraining', data_folder, dataset_name, model_name)

            # load data
            train = load_dataset(data_folder, dataset_name, 'train', logger=logger)

            # load fitted parameters
            fitted_params = load_parameters(data_folder, dataset_name, model_name, logger=logger)

            # stratified split to ensure validation data split does not cut off small classes
            train = pd.concat([*split_data(train, 0.1)])

            # retrain model
            model, history = retrain_model(train, models[model_name], fitted_params, logger=logger)

            # visualize retraining
            if history is not None:
                visualize_retraining(history, data_folder, dataset_name, model_name, logger=logger)

            # save trained model
            save_model(model, data_folder, dataset_name, model_name, logger=logger)
