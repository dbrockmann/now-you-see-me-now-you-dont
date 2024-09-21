
from sklearn.base import clone

from preprocessing import pipeline


def retrain_model(df, model, fitted_params, logger):
    """
    Retrain a model on the full training set with fitted parameters from randomized search
    """

    estimator = clone(pipeline).set_params(model=model, **fitted_params)

    logger.info('Starting model retraining.')
    
    estimator.fit(X=df.drop(columns='y'), y=df['y'])

    logger.info('Finished model retraining.')

    history = estimator['model'].history_ if hasattr(estimator['model'], 'history_') else None

    return estimator, history
