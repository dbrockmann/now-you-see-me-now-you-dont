
import numpy as np
import pandas as pd

from utils.visualization import visualize_class_shares, visualize_classification_report, visualize_confusion_matrix, visualize_shap_values


def evaluate_model(train, test, model, data_folder, dataset_name, model_name, logger):
    """
    Evaluate model
    """

    evaluate_class_counts(train, model, data_folder, dataset_name, model_name, logger)
    evaluate_feature_selection(model, logger)
    evaluate_test_set(test, model, data_folder, dataset_name, model_name, logger)

def evaluate_class_counts(train, model, data_folder, dataset_name, model_name, logger):
    """
    Evaluate the class counts from the training process, with the optimized under- and oversampling
    """

    class_counts = pd.DataFrame({'initial': train['y'].value_counts()})
    if model['undersampling'] is not None:
        class_counts['undersampling'] = pd.Series(model['undersampling'].sampling_strategy_)
        class_counts['undersampling'] = class_counts['undersampling'].fillna(class_counts['initial'])
    if model['oversampling'] is not None:
        class_counts['oversampling'] = pd.Series({k: model['oversampling'].sampling_strategy_[k] + (class_counts.loc[k, 'undersampling'] if 'undersampling' in class_counts else class_counts.loc[k, 'initial']) for k in class_counts.index})
        class_counts['oversampling'] = class_counts['oversampling'].fillna(class_counts['undersampling'] if not model['undersampling'] is None else class_counts['initial'])
    visualize_class_shares(class_counts, data_folder, dataset_name, model_name, logger)

def evaluate_feature_selection(model, logger):
    """
    Print the selected features from the feature selection
    """

    if model['feature_selection'] is None:
        logger.info('No feature selection applied.')
    else:
        sel_features = model['feature_selection'].get_feature_names_out()
        n_features_in = model['feature_selection'].n_features_in_

        logger.info('Following features are selected (%d/%d): %s.', len(sel_features), n_features_in, ', '.join(sel_features))

def evaluate_test_set(test, model, data_folder, dataset_name, model_name, logger):
    """
    Evaluate the model on the test set
    """

    predictions = model.predict(test.drop(columns='y'))
    classes = np.unique(test['y'])

    visualize_classification_report(test['y'], predictions, classes, data_folder, dataset_name, model_name, logger)
    visualize_confusion_matrix(test['y'], predictions, classes, data_folder, dataset_name, model_name, logger)

