
from parameters import datasets, models
from utils.io import get_logger
from utils.visualization import visualize_computational_performance


def plot_additionals(data_folder):
    """
    Plots additional visualizations
    """

    logger = get_logger('plot-additionals')
    logger.info('Starting plotting additionals')

    # training / test times in minutes extracted from log files
    comp_data = {
        "UNSW-NB15": {
            "DNN": {
                "train": 2.47,
                "test": 0.58
            },
            "CNN": {
                "train": 4.97,
                "test": 0.77
            },
            "AE": {
                "train": 6.08,
                "test": 0.57
            },
            "RF": {
                "train": 0.32,
                "test": 0.52
            },
            "SVM": {
                "train": 0.12,
                "test": 0.27
            },
            "KNN": {
                "train": 0.07,
                "test": 13.25
            }
        },
        "CIC-IDS2017-improved": {
            "DNN": {
                "train": 11.53,
                "test": 0.48
            },
            "CNN": {
                "train": 24.73,
                "test": 0.47
            },
            "AE": {
                "train": 45.9,
                "test": 0.45
            },
            "RF": {
                "train": 1.0,
                "test": 0.5
            },
            "SVM": {
                "train": 0.82,
                "test": 0.22
            },
            "KNN": {
                "train": 0.17,
                "test": 16.2
            }
        },
        "CSE-CIC-IDS2018-improved": {
            "DNN": {
                "train": 47.12,
                "test": 16.83
            },
            "CNN": {
                "train": 111.48,
                "test": 16.08
            },
            "AE": {
                "train": 208.02,
                "test": 17.03
            },
            "RF": {
                "train": 16.43,
                "test": 10.42
            },
            "SVM": {
                "train": 7.28,
                "test": 1.05
            },
            "KNN": {
                "train": 3.07,
                "test": 711.57
            }
        },
        "Web-IDS23": {
            "DNN": {
                "train": 42.57,
                "test": 3.1
            },
            "CNN": {
                "train": 44.68,
                "test": 2.8
            },
            "AE": {
                "train": 60.77,
                "test": 2.72
            },
            "RF": {
                "train": 5.37,
                "test": 2.47
            },
            "SVM": {
                "train": 1.38,
                "test": 1.35
            },
            "KNN": {
                "train": 0.68,
                "test": 282.18
            }
        }
    }

    visualize_computational_performance(comp_data, list(datasets), list(models), data_folder, logger)
