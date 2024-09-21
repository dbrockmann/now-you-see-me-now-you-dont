
import numpy as np

from .base import BaseAttack


class RandomInit(BaseAttack):
    """
    Attack to find adversarial examples in the constrained search space, using a simple iterative random sampling approach within a defined query budget
    """

    def __init__(self, is_adv_fn=None, features=None, features_min=None, features_max=None, features_discrete=None, features_lower=None, features_higher=None, features_base_change=None, feature_validation_fn=None, max_relative_perturbation=None, logger=None, max_iter=10):

        super().__init__(is_adv_fn=is_adv_fn, features=features, features_min=features_min, features_max=features_max, features_discrete=features_discrete, features_lower=features_lower, features_higher=features_higher, features_base_change=features_base_change, feature_validation_fn=feature_validation_fn, max_relative_perturbation=max_relative_perturbation, logger=logger)

        self.max_iter = max_iter

    def _generate(self, X):

        rng = np.random.default_rng(0)

        init_sample = np.copy(X)
        ind = np.arange(X.shape[0])

        # iterate in the query budget
        for i in range(self.max_iter):

            # sample random perturbation
            rand_sample = rng.random((ind.shape[0], *X.shape[1:]))

            is_adv = self._is_adv(rand_sample, ind)
            init_sample[ind[is_adv], :] = rand_sample[is_adv]
            ind = ind[~is_adv]

            if ind.size == 0:
                self.logger.info('All %d initial samples found.', X.shape[0])
                break

            if i == self.max_iter-1:
                self.logger.warning('%d/%d (%.2f%%) initial samples not found.', ind.size, X.shape[0], ind.size / X.shape[0] * 100)

        return init_sample
