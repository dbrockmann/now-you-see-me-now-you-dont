
import numpy as np

from .base import BaseAttackMin


class Pointwise(BaseAttackMin):
    """
    Pointwise attack: Lukas Schott, Jonas Rauber, Matthias Bethge, and Wieland Brendel. 2018. Towards the first adversarially robust neural network model on MNIST.
    Implementation in: https://github.com/bethgelab/foolbox

    We reimplement the attacks for our specific use case, but in most cases it is recommended to use the existing implementations in Foolbox (which we used as a reference as well)
    """

    def __init__(self, is_adv_fn=None, init_samples=None, features=None, features_min=None, features_max=None, features_discrete=None, features_lower=None, features_higher=None, features_base_change=None, feature_validation_fn=None, max_relative_perturbation=None, logger=None, binary_search_steps=10):
        """
        
        """

        super().__init__(is_adv_fn=is_adv_fn, init_samples=init_samples, features=features, features_min=features_min, features_max=features_max, features_discrete=features_discrete, features_lower=features_lower, features_higher=features_higher, features_base_change=features_base_change, feature_validation_fn=feature_validation_fn, max_relative_perturbation=max_relative_perturbation, logger=logger)

        self.binary_search_steps = binary_search_steps

    def _generate(self, X):
        """
        Generate adversarial examples from initial samples
        """

        rng = np.random.default_rng(0)

        self.logger.debug('Original: %s', X)
        self.logger.debug('Init: %s', self.init_samples_)

        # start with initial adversarial examples
        X_adv = np.copy(self.init_samples_)

        perturb_mask = np.ones_like(X_adv, dtype=bool)
        ind = np.arange(X_adv.shape[0])

        # first iteration through features to find features which can be completely removed from perturbation
        while ind.size > 0:

            f_ind = np.array([rng.permutation(p.nonzero()[0]) for p in perturb_mask[ind, :]], dtype=object)
            f_ind_num = np.array([f.size for f in f_ind])
            improved = np.zeros_like(ind, dtype=bool)

            for i in range(np.max(f_ind_num)):

                f_ind_ = np.array([f[i] for f in f_ind[i < f_ind_num]])
                ind_ = ind[i < f_ind_num]

                perturb_mask_ = perturb_mask[ind_, :]
                perturb_mask_[np.arange(perturb_mask_.shape[0]), f_ind_] = False

                X_adv_ = X_adv[ind_, :]
                X_adv_[~perturb_mask_] = X[ind_, :][~perturb_mask_]

                is_adv = self._is_adv(X_adv_, ind=ind_)

                perturb_mask[ind_, f_ind_] = ~is_adv
                improved[i < f_ind_num] = np.where(is_adv, True, improved[i < f_ind_num])

                self.logger.debug('Perturb mask: %s', perturb_mask)

            ind = ind[improved]


        ind = np.arange(X_adv.shape[0])

        # second iteration with binary search to decrease feature distance
        while ind.size > 0:

            f_ind = np.array([rng.permutation(p.nonzero()[0]) for p in perturb_mask[ind, :]], dtype=object)
            f_ind_num = np.array([f.size for f in f_ind])
            improved = np.zeros_like(ind, dtype=bool)

            for i in range(np.max(f_ind_num)):

                f_ind_ = np.array([f[i] for f in f_ind[i < f_ind_num]])
                ind_ = ind[i < f_ind_num]

                perturb_mask_ = perturb_mask[ind_, :]
                perturb_mask_[np.arange(perturb_mask_.shape[0]), f_ind_] = False

                X_adv_ = X_adv[ind_, :]
                X_adv_[~perturb_mask_] = X[ind_, :][~perturb_mask_]
                is_adv = self._is_adv(X_adv_, ind=ind_)
                perturb_mask[ind_, f_ind_] = ~is_adv
                improved_ = np.copy(is_adv)

                if np.any(~is_adv):
                    X_adv_ = X_adv[ind_, :]
                    X_adv_[~perturb_mask[ind_, :]] = X[ind_, :][~perturb_mask[ind_, :]]
                    X_adv[ind_[~is_adv], :], improved_[~is_adv] = self._binary_search(X_adv_[~is_adv, :], X[ind_[~is_adv], :], f_ind_[~is_adv], ind_[~is_adv])

                improved[i < f_ind_num] = np.where(improved_, True, improved[i < f_ind_num])

            ind = ind[improved]


        X_adv[~perturb_mask] = X[~perturb_mask]
        
        return X_adv

    def _binary_search(self, X_adv, X, f_ind, ind):
        """
        Perform binary search, following the official implementation of the paper in Foolbox
        """

        lower = np.full(X.shape[0], 0.0)
        upper = np.full(X.shape[0], 1.0)

        for _ in range(self.binary_search_steps):

            alpha = (lower + upper) / 2.0
            is_adv = self._is_adv(self._project(X_adv, X, f_ind, alpha), ind=ind)

            lower = np.where(is_adv, lower, alpha)
            upper = np.where(is_adv, alpha, upper)

        X_bound = self._project(X_adv, X, f_ind, upper)
        improved = np.abs(X_bound[np.arange(X.shape[0]), f_ind] - X_adv[np.arange(X.shape[0]), f_ind]) > 1e-15 # fixes numerical precision error if using upper < 1

        X_bound[~improved, :] = X_adv[~improved, :]

        return X_bound, improved
    
    def _project(self, X_adv, X, f_ind, alpha):
        """
        Project between adversarial and original for a given feature
        """

        alpha_ = np.ones_like(X, dtype=float)
        alpha_[np.arange(X.shape[0]), f_ind] = alpha

        projected = np.clip(alpha_ * X_adv + (1 - alpha_) * X, 0.0, 1.0)

        return projected
    