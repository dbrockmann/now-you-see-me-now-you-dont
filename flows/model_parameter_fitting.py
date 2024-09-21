
from parameters import datasets, models, model_selection, model_parameters
from classification import fit_parameters
from utils.io import get_logger, load_dataset, load_parameters, save_parameters


def model_parameter_fitting(data_folder):
    """
    Fit the parameters for the models using randomized search with cross validation
    """

    logger_flow = get_logger('model-parameter-fitting').info('Starting parameter fitting for the following datasets: %s; and the following models: %s.', ', '.join(datasets.keys()), ', '.join(models.keys()))

    for dataset_name in datasets:

        # load data
        train = load_dataset(data_folder, dataset_name, 'train', logger=logger_flow)

        for model_name in models:

            logger = get_logger('model-parameter-fitting', data_folder, dataset_name, model_name)

            # load fitted parameters
            loaded_params = load_parameters(data_folder, dataset_name, model_name, logger=logger)
            fitted_params = {'undersampling': loaded_params['undersampling'], 'feature_selection': loaded_params['feature_selection'], 'oversampling': loaded_params['oversampling']}

            # fit parameters
            new_params = fit_parameters(train, models[model_name], fitted_params, model_parameters[model_name], logger, n_iter=model_selection['n_iter'], n_folds=model_selection['n_folds'], n_jobs=model_selection['n_jobs'])

            # save fitted parameters
            save_parameters(fitted_params | new_params, data_folder, dataset_name, model_name, logger=logger)
