
import numpy as np

from .base import BaseAttackMin


class BaselineMin(BaseAttackMin):
    """
    Baseline for minimization attacks: returns initial samples without active minimization
    """

    def __init__(self, is_adv_fn=None, init_samples=None, features=None, features_min=None, features_max=None, features_discrete=None, features_lower=None, features_higher=None, features_base_change=None, feature_validation_fn=None, max_relative_perturbation=None, logger=None):

        super().__init__(is_adv_fn=is_adv_fn, init_samples=init_samples, features=features, features_min=features_min, features_max=features_max, features_discrete=features_discrete, features_lower=features_lower, features_higher=features_higher, features_base_change=features_base_change, feature_validation_fn=feature_validation_fn, max_relative_perturbation=max_relative_perturbation, logger=logger)

    def _generate(self, X):

        return np.copy(self.init_samples_)
    