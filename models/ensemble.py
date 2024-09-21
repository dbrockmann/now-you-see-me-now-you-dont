
import numpy as np


class EnsembleModel:
    """
    Ensemble model using a voting mechanism
    """

    def __init__(self, model_names, models=[]):

        self.model_names = model_names
        self.models = models

    def predict(self, input):

        predictions = [model.predict(input) for model in self.models]
        ens_predictions = np.array([self._voting(np.array(pred).flatten()) for pred in zip(*predictions)])
        
        return ens_predictions
        
    def _voting(self, predictions):
        """
        Voting mechanism, predict benign only if all submodels predict benign. Otherwise predict most voted class.
        """

        if np.all(predictions == 'Benign'):
            return 'Benign'
        else:
            filt_preds = predictions[predictions != 'Benign']
            unique, counts = np.unique(filt_preds, return_counts=True)
            return unique[np.argmax(counts)]
