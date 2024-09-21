
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


class CustomRandomUnderSampler(RandomUnderSampler):
    """
    Fixes difficulties with the undersampler by preserving the original (shuffled) order instead of sorted by class. 
    This is especially important for early stopping validation splits, which would otherwise cut off entire classes.
    """

    def __init__(self, steps, step, threshold, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.step = step
        self.threshold = threshold

    def fit_resample(self, X, y):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        self.sampling_strategy = undersampling_strategy(*np.unique(y, return_counts=True), self.steps, self.step, self.threshold)
        X, y = super().fit_resample(X, y)
        X = X.sort_index()
        y = y.sort_index()

        return X, y
    
def undersampling_strategy(classes, counts, steps, step, threshold):
    """
    Undersampling strategy
    """
    
    min_samples = max(counts[counts <= threshold * np.sum(counts)])
    strategy = {class_name: min(int(count), np.geomspace(count, min_samples, num=steps, dtype=int)[step]) for class_name, count in zip(classes, counts)}

    return strategy

class CustomSMOTE(SMOTE):
    """
    Fixes difficulties with the oversampler by randomly inserting the additional samples instead of appending them to the end.
    """

    def __init__(self, steps, step_class, max_samples_class, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.step_class = step_class
        self.max_samples_class = max_samples_class

    def fit_resample(self, X, y):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        self.sampling_strategy = oversampling_strategy(*np.unique(y, return_counts=True), self.steps, self.step_class, self.max_samples_class)
        X_resampled, y_resampled = super().fit_resample(X, y)

        rand_ind = np.random.default_rng(0).choice(len(y_resampled.index), len(y_resampled.index)-len(y.index), replace=False)
        new_index = np.concatenate((np.setdiff1d(np.arange(len(y_resampled.index)), rand_ind), rand_ind))
        X_resampled.index = new_index
        y_resampled.index = new_index
        X_resampled = X_resampled.sort_index()
        y_resampled = y_resampled.sort_index()

        return X_resampled, y_resampled
    
def oversampling_strategy(classes, counts, steps, step_class, max_samples_class):
    """
    Oversampling strategy
    """

    strategy = {class_name: int(count) if class_name not in step_class else np.geomspace(count, counts[classes==max_samples_class][0], num=steps, dtype=int)[step_class[class_name]] for class_name, count in zip(classes, counts)}

    return strategy
