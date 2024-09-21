
from parameters import datasets
from preprocessing import clean_data
from datasets import BinaryDataset
from utils.data import split_data
from utils.io import get_logger, save_dataset


def data_cleaning(data_folder):
    """
    Loads the datasets defined by the parameters, applies the data cleaning pipeline, and finally saves the datasets.
    """

    non_bin_datasets = [dataset_name for dataset_name in datasets if not isinstance(datasets[dataset_name], BinaryDataset)]
    get_logger('data-cleaning').info('Starting data cleaning for the following datasets: %s.', ', '.join(non_bin_datasets))

    for dataset_name in non_bin_datasets:

        logger = get_logger('data-cleaning', data_folder, dataset_name)

        # load data
        data = datasets[dataset_name].load(logger=logger)

        # clean data
        data = clean_data(data, data_folder=data_folder, dataset_name=dataset_name, logger=logger)

        # split data and safe
        train, test = split_data(data, split=0.4, logger=logger)
        save_dataset(train, data_folder, dataset_name, 'train', logger=logger)
        save_dataset(test, data_folder, dataset_name, 'test', logger=logger)
