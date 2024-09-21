
import numpy as np
from sklearn.base import clone


def initialize_samples(df, model, attack_init, features, features_min, features_max, features_discrete, features_lower, features_higher, features_base_change, feature_validation_fn, logger, binary_classifier=False):
    """
    Create adversarial examples from correct attack predictions
    """

    clean_preds = model.predict(df.drop(columns='y'))
    if binary_classifier:
        samples = df[(clean_preds == 'Attack') & (df['y'] != 'Benign')].reset_index(drop=True)
    else:
        samples = df[(clean_preds == df['y']) & (df['y'] != 'Benign')].reset_index(drop=True)

    attack_init = clone(attack_init).set_params(
        is_adv_fn = lambda X: model.predict(X) == 'Benign', 
        features = features,
        features_min = features_min,
        features_max = features_max,
        features_discrete = features_discrete,
        features_lower = features_lower,
        features_higher = features_higher,
        features_base_change = features_base_change,
        feature_validation_fn = feature_validation_fn,
        logger = logger
    )

    samples_adv = attack_init.fit_transform(samples.drop(columns='y'))
    samples_adv['y'] = model.predict(samples_adv)
    samples_adv['query_count'] = attack_init.query_counts_

    is_adv = samples_adv['y'] == 'Benign'
    logger.info('Successfully initialized %d/%d (%.2f%%) adversarial examples with an average of %.2f queries (%.2f including unsuccessful attempts).', np.sum(is_adv), len(is_adv), np.sum(is_adv) / len(is_adv) * 100, np.mean(attack_init.query_counts_[is_adv]), np.mean(attack_init.query_counts_))

    return samples, samples_adv

def minimize_samples(original_samples, init_samples, model, attack_min, features, features_min, features_max, features_discrete, features_lower, features_higher, features_base_change, feature_validation_fn, logger):
    """
    Minimize successful adversarial examples
    """

    successful_inits = (original_samples['y'] != 'Benign') & (init_samples['y'] == 'Benign')
    samples = original_samples.loc[successful_inits, :].reset_index(drop=True)
    init_samples = init_samples.loc[successful_inits, :].reset_index(drop=True)

    attack_min = clone(attack_min).set_params(
        is_adv_fn = lambda X: model.predict(X) == 'Benign', 
        init_samples = init_samples.drop(columns=['y', 'query_count']),
        features = features,
        features_min = features_min,
        features_max = features_max,
        features_discrete = features_discrete,
        features_lower = features_lower,
        features_higher = features_higher,
        features_base_change = features_base_change,
        feature_validation_fn = feature_validation_fn,
        logger = logger,
    )

    samples_adv = attack_min.fit_transform(samples.drop(columns='y'))
    samples_adv['y'] = model.predict(samples_adv)
    samples_adv['query_count'] = attack_min.query_counts_

    is_adv = samples_adv['y'] == 'Benign'
    not_adv_count = np.sum(~is_adv)
    if not_adv_count > 0:
        logger.error('%d samples are not adversarial anymore due to an unknown error.', not_adv_count)
    assert not_adv_count == 0
    
    logger.info('Minimized %d samples with a mean of %.2f queries.', len(samples_adv.index), np.mean(attack_min.query_counts_))

    return samples, samples_adv
