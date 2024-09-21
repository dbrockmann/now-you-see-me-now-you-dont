
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BaseAttack(BaseEstimator, TransformerMixin):
    """
    Base class for an attack, applies the correct transformations for the constrained search space
    """

    def __init__(self, is_adv_fn=None, features=None, features_min=None, features_max=None, features_discrete=None, features_lower=None, features_higher=None, features_base_change=None, feature_validation_fn=None, max_relative_perturbation=None, logger=None):
        self.is_adv_fn = is_adv_fn
        self.features = features
        self.features_min = features_min
        self.features_max = features_max
        self.features_discrete = features_discrete
        self.features_lower = features_lower
        self.features_higher = features_higher
        self.features_base_change = features_base_change
        self.feature_validation_fn = feature_validation_fn
        self.max_relative_perturbation = max_relative_perturbation
        self.logger = logger

    def fit(self, X, y=None):
        """
        Initialize variables and search space
        """

        self.X_ = None
        self.query_counts_ = None
        self.features_min_ = np.array(self.features_min)
        self.features_max_ = np.array(self.features_max)

        # set technical min/max values
        if len(self.features_min_.shape) == 1:
            self.features_min_ = np.full((len(X.index), len(self.features)), self.features_min_)
        if len(self.features_max_.shape) == 1:
            self.features_max_ = np.full((len(X.index), len(self.features)), self.features_max_)

        # set boundaries from relative max perturbation
        if self.max_relative_perturbation is not None:
            self.features_min_ = X.loc[:, self.features].to_numpy() * (1 - self.max_relative_perturbation)
            self.features_max_ = X.loc[:, self.features].to_numpy() * (1 + self.max_relative_perturbation)

        # apply lower/higher restrictions
        not_lower = ~np.array(self.features_lower)
        not_higher = ~np.array(self.features_higher)
        self.features_min_[:, not_lower] = np.clip(self.features_min_[:, not_lower], X.loc[:, np.array(self.features)[not_lower]].to_numpy() + np.array(self.features_base_change)[not_lower], None)
        self.features_max_[:, not_higher] = np.clip(self.features_max_[:, not_higher], None, X.loc[:, np.array(self.features)[not_higher]].to_numpy() - np.array(self.features_base_change)[not_higher])

        # apply rounding of boundaries for discrete features
        self.features_min_[:, self.features_discrete] = np.round(self.features_min_[:, self.features_discrete])
        self.features_max_[:, self.features_discrete] = np.round(self.features_max_[:, self.features_discrete])
        self.features_min_ = np.clip(self.features_min_, a_min=np.array(self.features_min), a_max=np.array(self.features_max))
        self.features_max_ = np.clip(self.features_max_, a_min=np.array(self.features_min), a_max=np.array(self.features_max))
        
        self.X_ = X.reset_index(drop=True)
        self.query_counts_ = np.full(len(self.X_.index), 0)

        assert np.all(self.features_min_ <= self.features_max_)
        assert ((len(self.X_.index), len(self.features)) == self.features_min_.shape == self.features_max_.shape)

        return self
    
    def transform(self, X, y=None):
        """
        Generate adversarial examples from original attack samples by normalizing, 
        applying the attack, inverse normalizing and enforcing constraints between features
        """

        X_f = self._normalize(self.X_.loc[:, self.features].to_numpy())

        X_adv = self.X_.copy()
        X_adv.loc[:, self.features] = self._inverse_normalize(self._generate(X_f))
        X_adv = self.feature_validation_fn(self.X_, X_adv)

        return X_adv
    
    def _normalize(self, X):
        """
        Normalize features to the min/max boundaries
        """

        X_norm = np.where(self.features_max_ > self.features_min_, (X - self.features_min_) / np.clip(self.features_max_ - self.features_min_, a_min=1e-15, a_max=None), 0.0)
        X_norm = np.clip(X_norm, 0.0, 1.0)

        return X_norm
    
    def _inverse_normalize(self, X, ind=None):
        """
        Apply the inverse normalization to the original feature space
        """

        if ind is None:
            ind = np.arange(len(self.X_.index))
        assert (X.shape[0] == len(ind))

        X_invnorm = np.clip(X, 0.0, 1.0) * (self.features_max_[ind] - self.features_min_[ind]) + self.features_min_[ind]
        X_invnorm[:, self.features_discrete] = np.round(X_invnorm[:, self.features_discrete])
        X_invnorm = np.clip(X_invnorm, self.features_min_[ind], self.features_max_[ind])

        return X_invnorm

    def _is_adv(self, X, ind=None):
        """
        Query the black-box victim model and check if samples are adversary
        """

        if ind is None:
            ind = np.arange(len(self.X_.index))
        assert (X.shape[:-1] == ind.shape)
        ind = ind.flatten()

        self.query_counts_[np.unique(ind)] += np.unique(ind, return_counts=True)[1]

        X_adv = self.X_.loc[ind]
        X_adv.loc[:, self.features] = self._inverse_normalize(X.reshape(-1, len(self.features)), ind=ind)
        X_adv = self.feature_validation_fn(self.X_.loc[ind], X_adv)

        is_adv = np.array(self.is_adv_fn(X_adv)).reshape(*X.shape[:-1])

        return is_adv

    def _generate(self, X):
        """
        Abstract.
        """

        raise NotImplementedError

class BaseAttackMin(BaseAttack):
    """
    Base class for minimization attacks, with initial samples
    """

    def __init__(self, is_adv_fn=None, init_samples=None, features=None, features_min=None, features_max=None, features_discrete=None, features_lower=None, features_higher=None, features_base_change=None, feature_validation_fn=None, max_relative_perturbation=None, logger=None):

        super().__init__(is_adv_fn=is_adv_fn, features=features, features_min=features_min, features_max=features_max, features_discrete=features_discrete, features_lower=features_lower, features_higher=features_higher, features_base_change=features_base_change, feature_validation_fn=feature_validation_fn, max_relative_perturbation=max_relative_perturbation, logger=logger)

        self.init_samples = init_samples

    def fit(self, X, y=None):

        super().fit(X, y)

        self.init_samples_ = self._normalize(self.init_samples.loc[:, self.features].to_numpy())

        return self
    