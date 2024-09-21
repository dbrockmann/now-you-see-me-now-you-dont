
import numpy as np

from .base import BaseAttackMin


class Boundary(BaseAttackMin):
    """
    Boundary attack: Wieland Brendel, Jonas Rauber, and Matthias Bethge. 2018. Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models.
    Implementation in: https://github.com/bethgelab/foolbox

    We reimplement the attacks for our specific use case, but in most cases it is recommended to use the existing implementations in Foolbox (which we used as a reference as well)
    """

    def __init__(self, is_adv_fn=None, init_samples=None, features=None, features_min=None, features_max=None, features_discrete=None, features_lower=None, features_higher=None, features_base_change=None, feature_validation_fn=None, max_relative_perturbation=None, logger=None, max_iter=1000, init_orth_step_size=0.01, init_orig_step_size=0.01, step_size_update=2, step_size_update_interval=1, orth_step_hist_size=50, orig_step_hist_size=30, orig_step_size_convergence=1e-5):

        super().__init__(is_adv_fn=is_adv_fn, init_samples=init_samples, features=features, features_min=features_min, features_max=features_max, features_discrete=features_discrete, features_lower=features_lower, features_higher=features_higher, features_base_change=features_base_change, feature_validation_fn=feature_validation_fn, max_relative_perturbation=max_relative_perturbation, logger=logger)

        self.max_iter = max_iter
        self.init_orth_step_size = init_orth_step_size
        self.init_orig_step_size = init_orig_step_size
        self.step_size_update = step_size_update
        self.step_size_update_interval = step_size_update_interval
        self.orth_step_hist_size = orth_step_hist_size
        self.orig_step_hist_size = orig_step_hist_size
        self.orig_step_size_convergence = orig_step_size_convergence

    def _generate(self, X):
        """
        Generate adversarial examples from initial samples
        """

        self.logger.debug('Original: %s', X)
        self.logger.debug('Init: %s', self.init_samples_)

        # start with initial adversarial examples
        X_adv = self.init_samples_

        # initialize step sizes
        orth_step_size = np.full(X.shape[0], self.init_orth_step_size)
        orig_step_size = np.full(X.shape[0], self.init_orig_step_size)

        # track adversarial stats
        orth_step_hist = np.full((X.shape[0], self.orth_step_hist_size), np.nan)
        orig_step_hist = np.full((X.shape[0], self.orig_step_hist_size), np.nan)

        ind = np.arange(X.shape[0])
        
        for k in range(0, self.max_iter):

            if ind.size == 0:
                self.logger.info('Aborting early after %d/%d iterations. All samples converged.', k, self.max_iter)
                break

            # add perturbations from proposal distribution
            orth_candidate, orig_candidate, closer = self._draw_candidates(X_adv[ind], X[ind], orth_step_size, orig_step_size)

            # check if sample is adversarial with added perturbation
            is_adv = self._is_adv(orig_candidate, ind=ind)
            self.logger.debug('Is Adversarial: %f', np.mean(is_adv))

            # keep perturbation for adversarial samples
            X_adv[ind[is_adv & closer]] = orig_candidate[is_adv & closer]

            # update step sizes
            if (k + 1) % self.step_size_update_interval == 0:

                self.logger.debug('Step: %s', k)
                #self.logger.debug('Orth Step: %s', orth_candidate)
                #self.logger.debug('Orig Step: %s', orig_candidate)
                #self.logger.debug('Updated Samples: %s', X_adv[ind])

                update_num = k // self.step_size_update_interval

                # update history
                orth_step_hist[:, update_num % self.orth_step_hist_size] = self._is_adv(orth_candidate, ind=ind)
                orig_step_hist[:, update_num % self.orig_step_hist_size] = is_adv

                # update step size based on orth step
                hist_full = ~np.isnan(orth_step_hist).any(axis=-1)
                adv_ratio = np.mean(orth_step_hist, axis=-1)

                cond_high_adv = hist_full & (adv_ratio > 0.5)
                orth_step_size[cond_high_adv] *= self.step_size_update
                orig_step_size[cond_high_adv] *= self.step_size_update

                cond_low_adv = hist_full & (adv_ratio < 0.2)
                orth_step_size[cond_low_adv] /= self.step_size_update
                orig_step_size[cond_low_adv] /= self.step_size_update

                orth_step_hist[cond_high_adv | cond_low_adv] = np.nan

                # update step size based on orig step
                hist_full = ~np.isnan(orig_step_hist).any(axis=-1)
                adv_ratio = np.mean(orig_step_hist, axis=-1)

                cond_high_adv = hist_full & (adv_ratio > 0.25)
                orig_step_size[cond_high_adv] *= self.step_size_update

                cond_low_adv = hist_full & (adv_ratio < 0.1)
                orig_step_size[cond_low_adv] /= self.step_size_update

                orig_step_hist[cond_high_adv | cond_low_adv] = np.nan

                #self.logger.debug('Orth step size: %s', orth_step_size)
                #self.logger.debug('Orig step size: %s', orig_step_size)

                converged = (orig_step_size < self.orig_step_size_convergence) | (orig_step_size > 1) # extra condition for always adversarial sample, which might rarely happen due to data inconsistencies regarding the constraints
                ind = ind[~converged]
                orth_step_size = orth_step_size[~converged]
                orig_step_size = orig_step_size[~converged]
                orth_step_hist = orth_step_hist[~converged]
                orig_step_hist = orig_step_hist[~converged]
        
        if ind.size > 0:
            self.logger.warning('%d/%d (%.2f%%) samples did not converge.', ind.size, X.shape[0], ind.size / X.shape[0] * 100)

        return X_adv

    def _draw_candidates(self, X_adv, X, orth_step_size, orig_step_size):
        """
        Draw candidates for both steps
        """

        rng = np.random.default_rng(0)

        direction = X - X_adv
        direction_norm = np.linalg.norm(direction, ord=2, axis=-1)
        direction_normed = direction / (direction_norm + 1e-12).reshape(-1, 1)

        perturbation = rng.normal(size=X_adv.shape[1:])
        perturbation = perturbation.T - np.matmul(direction_normed, perturbation).reshape(-1, 1) * direction_normed
        perturbation = perturbation * (orth_step_size * direction_norm / np.linalg.norm(perturbation, ord=2, axis=-1)).reshape(-1, 1)
        
        orth_candidate = np.clip(X + (perturbation - direction) / np.sqrt(np.square(orth_step_size) + 1).reshape(-1, 1), 0, 1)

        new_direction = X - orth_candidate
        new_direction_norm = np.linalg.norm(new_direction, ord=2, axis=-1)

        length = np.clip(orig_step_size * direction_norm + new_direction_norm - direction_norm, a_min=0, a_max=None) / (new_direction_norm + 1e-12)

        orig_candidate = np.clip(orth_candidate + length.reshape(-1, 1) * new_direction, 0, 1)

        # fix limited numerical precision
        closer = np.linalg.norm(X - orig_candidate, ord=2, axis=-1) < direction_norm

        return orth_candidate, orig_candidate, closer
