
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV, train_test_split, PredefinedSplit
from sklearn.metrics import make_scorer, f1_score

from preprocessing import pipeline


def fit_parameters(df, model, params, test_params, logger, n_iter=20, n_folds=5, n_jobs=1):
    """
    Fit parameters using randomized search
    """

    search_cv = RandomizedSearchCV(
        estimator = clone(pipeline).set_params(model=model, **params),
        n_iter = n_iter,
        cv = StratifiedKFold(n_splits=n_folds, shuffle=False),
        n_jobs = n_jobs,
        scoring = 'f1_macro',
        refit = False,
        param_distributions = test_params,
        verbose = 2,
    )

    logger.info('Starting parameter fitting with %d iterations, %d-fold cross validation and %d jobs.', n_iter, n_folds, n_jobs)
    search_cv.fit(X=df.drop(columns='y'), y=df['y'])

    results = search_cv.cv_results_
    best_index = np.argmin(results['rank_test_score'])
    best_score = results['mean_test_score'][best_index]
    best_params = results['params'][best_index]

    logger.info('Finished parameter fitting. The best set of parameters achieved a macro F1 score of %.4f. Following parameters were selected: %s.', best_score, ', '.join([f'{k}: {v}' for k, v in best_params.items()]))

    return best_params


def test_parameters(df, model, params, test_params, logger, labels=None, n_folds=5, n_jobs=1):
    """
    Test specific parameters using grid search with cross validation
    """

    scoring = {'f1': 'f1_macro'}
    if labels is not None:
        scoring['f1_labels'] = make_scorer(f1_score, labels=labels, average='macro')

    search_cv = GridSearchCV(
        estimator = clone(pipeline).set_params(model=model, **params),
        cv = StratifiedKFold(n_splits=n_folds, shuffle=False),
        n_jobs = n_jobs,
        scoring = scoring,
        refit = False,
        param_grid = test_params,
        return_train_score = True,
        verbose = 2,
    )

    logger.info('Starting parameter testing with %d-fold cross validation and %d jobs.', n_folds, n_jobs)
    search_cv.fit(X=df.drop(columns='y'), y=df['y'])

    results = search_cv.cv_results_

    logger.info('Finished parameter testing.')

    return results

def test_parameters_simple(df, model, params, test_params, logger, labels=None, n_jobs=1):
    """
    Test specific parameters using grid search with a simple train/test split
    """

    scoring = {'f1': 'f1_macro'}
    if labels is not None:
        scoring['f1_labels'] = make_scorer(f1_score, labels=labels, average='macro')

    train_ind, test_ind = train_test_split(np.arange(len(df.index)), test_size=0.2, shuffle=True, stratify=df['y'], random_state=0)

    search_cv = GridSearchCV(
        estimator = clone(pipeline).set_params(model=model, **params),
        cv = [(train_ind, test_ind)],
        n_jobs = n_jobs,
        scoring = scoring,
        refit = False,
        param_grid = test_params,
        return_train_score = False,
        verbose = 2,
    )

    logger.info('Starting simple parameter testing with %d jobs.', n_jobs)
    search_cv.fit(X=df.drop(columns='y'), y=df['y'])
    logger.info('Grid search finished.')

    results = search_cv.cv_results_

    logger.info('Finished parameter testing.')

    return results

    

