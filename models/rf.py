
from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(
    n_estimators = 100,
    criterion = 'entropy',
    max_depth = 15,
    random_state = 0,
    verbose = 0,
)
