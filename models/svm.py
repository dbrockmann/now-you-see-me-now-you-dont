
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem


model = Pipeline([
    ('kernel', Nystroem(kernel='rbf', n_components=100, gamma=0.1, kernel_params = {'C': 1}, random_state=0)),
    ('svm', SGDClassifier(alpha=0.00001, class_weight='balanced', loss='hinge', max_iter=10000, random_state=0))
])
