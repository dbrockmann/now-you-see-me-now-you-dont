
import numpy as np

from .base import BaseAttackMin


class HopSkipJump(BaseAttackMin):
    """
    HopSkipJump attack: Jianbo Chen, Michael I. Jordan, and Martin J. Wainwright. 2020. HopSkipJumpAttack: A Query-Efficient Decision-Based Attack.
    Implementation in: https://github.com/bethgelab/foolbox

    We reimplement the attacks for our specific use case, but in most cases it is recommended to use the existing implementations in Foolbox (which we used as a reference as well)
    """

    def __init__(self, is_adv_fn=None, init_samples=None, features=None, features_min=None, features_max=None, features_discrete=None, features_lower=None, features_higher=None, features_base_change=None, feature_validation_fn=None, max_relative_perturbation=None, logger=None, iter=16, init_batch_size=16, norm=2):

        super().__init__(is_adv_fn=is_adv_fn, init_samples=init_samples, features=features, features_min=features_min, features_max=features_max, features_discrete=features_discrete, features_lower=features_lower, features_higher=features_higher, features_base_change=features_base_change, feature_validation_fn=feature_validation_fn, max_relative_perturbation=max_relative_perturbation, logger=logger)

        self.iter = iter
        self.init_batch_size = init_batch_size
        self.norm = norm

    def _generate(self, X):
        """
        Generate adversarial examples from initial samples
        """

        self.logger.debug('Original: %s', X)

        self.logger.debug('Init: %s', self.init_samples_)

        # start with initial adversarial examples
        X_adv = self.init_samples_
        
        # compute dimension (d)
        dimension = int(np.prod(X.shape[1:]))

        # set binary search threshold (theta) according to equation (15)
        threshold = np.power(dimension, 1 / self.norm - 2)
        threshold = 0.001 # fix for very small dimensions

        self.logger.debug('Threshold: %f', threshold)

        for t in range(1, self.iter):


            # BOUNDARY SEARCH

            # perform binary search to approach the decision boundary
            X_bound = self._binary_search(X_adv, X, threshold)

            self.logger.debug('Boundary: %s', X_bound)


            # GRADIENT-DIRECTION ESTIMATION

            # compute batch size (B_t)
            batch_size = int(self.init_batch_size * np.sqrt(t))

            # compute perturbation size (delta_t) according to equation (15)
            perturb_size = 1 / dimension * self._distance(X_adv, X)

            self.logger.debug('Perturb Size: %s', perturb_size)

            # estimate gradient-direction (S Delta) according to equation (16)
            gradient_estimate = self._estimate_gradient(X_bound, batch_size, perturb_size)

            self.logger.debug('Gradient Estimate: %s', gradient_estimate)

            # compute update (v_t) according to equation (12)
            update = self._compute_update(gradient_estimate)

            self.logger.debug('Update: %s', update)


            # STEP SIZE SEARCH

            # set initial step size
            init_step_size = self._distance(X_bound, X) / np.sqrt(t)

            # search step size through geometric progression
            step_size = self._search_step_size(X_bound, update, init_step_size)


            # perform update
            X_adv = np.clip(X_bound + step_size.reshape(-1, 1) * update, 0.0, 1.0)

            self.logger.debug('Updated: %s', X_adv)

        # final binary search
        X_bound = self._binary_search(X_adv, X, threshold)

        self.logger.debug('Final Boundary: %s', X_bound)

        return X_bound
    
    def _distance(self, X_adv, X):
        """
        Calculate distance based on norm
        """

        distance = np.linalg.norm(X_adv - X, ord=self.norm, axis=-1)

        return distance
    
    def _binary_search(self, X_adv, X, threshold):
        """
        Perform binary search to find decision boundary
        """

        lower = np.full(X.shape[0], 0.0)
        upper = np.full(X.shape[0], 1.0)

        while np.abs(lower[0] - upper[0]) >= threshold:

            alpha = (lower + upper) / 2.0
            is_adv = self._is_adv(self._project(X_adv, X, alpha))

            lower = np.where(is_adv, lower, alpha)
            upper = np.where(is_adv, alpha, upper)

        self.logger.debug('Bounds: %s', upper)

        X_bound = self._project(X_adv, X, upper)

        return X_bound

    def _project(self, X_adv, X, alpha):
        """
        Project between adversarial and original samples
        """

        if self.norm == 2: # we swap X_adv and X, which seems to be wrong in the paper
            projection = alpha.reshape(-1, 1) * X_adv + (1 - alpha).reshape(-1, 1) * X

        elif self.norm == np.inf:
            c = (alpha * self._distance(X_adv, X)).reshape(-1, 1)
            projection = np.max([np.min([X, X + c], axis=0), X_adv - c], axis=0)

        return projection

    def _estimate_gradient(self, X_bound, batch_size, perturb_size):
        """
        Perform gradient estimation by sampling
        """

        rng = np.random.default_rng(0)

        # sample unit vectors
        noise = rng.normal(size=(batch_size, *X_bound.shape[1:]))
        unit_vectors = noise / (np.linalg.norm(noise, ord=2, axis=-1) + 1e-12).reshape(batch_size, 1)

        # apply perturbations
        X_perturbed = np.clip(X_bound.reshape(1, *X_bound.shape) + perturb_size.reshape(1, -1, 1) * unit_vectors.reshape(batch_size, 1, *X_bound.shape[1:]), 0.0, 1.0)
        unit_vectors_upd = (X_perturbed - X_bound.reshape(1, *X_bound.shape)) / (perturb_size.reshape(1, -1, 1) + 1e-12) # update unit vectors to clip operation

        # query victim model with perturbed samples
        is_adv = np.where(self._is_adv(X_perturbed, ind=np.resize(np.arange(X_bound.shape[0]), batch_size*X_bound.shape[0]).reshape(batch_size, -1)), 1.0, -1.0)

        # compute gradient estimation
        baseline = np.mean(is_adv, axis=0)
        estimate = 1 / (batch_size - 1) * np.sum((is_adv - baseline.reshape(1, -1)).reshape(batch_size, -1, 1) * unit_vectors_upd, axis=0)

        return estimate
    
    def _compute_update(self, gradient_estimate):
        """
        Compute update from the gradient estimations
        """

        if self.norm == 2:

            update = gradient_estimate / (np.linalg.norm(gradient_estimate, ord=2, axis=-1) + 1e-12).reshape(*gradient_estimate.shape[:-1], 1)
        
        else:

            update = np.sign(gradient_estimate)

        return update
    
    def _search_step_size(self, X_bound, update, init_step_size):
        """
        Search for the step size using geometric progression
        """

        step_size = np.copy(init_step_size)
        ind = np.arange(X_bound.shape[0])

        while ind.size > 0 and np.all(step_size[ind] > 1e-15):

            ind = ind[~self._is_adv(np.clip(X_bound[ind] + step_size[ind].reshape(-1, 1) * update[ind], 0.0, 1.0), ind=ind)]
            step_size[ind] = step_size[ind] / 2.0

        step_size[step_size < 1e+15] = 0.0 # fix low precision error

        return step_size
    