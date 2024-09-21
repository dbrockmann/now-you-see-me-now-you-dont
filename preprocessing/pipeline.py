
from imblearn.pipeline import Pipeline as Pipeline_with_sampler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer


# processing pipeline
pipeline = Pipeline_with_sampler([
    ('undersampling', None),
    ('feature_selection', None),
    ('encoding', ColumnTransformer([
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), make_column_selector(dtype_include='category')),
        ('normalization', QuantileTransformer(n_quantiles=1000, random_state=0), make_column_selector(dtype_include='number'))
    ], remainder='drop', verbose_feature_names_out=False).set_output(transform='pandas')),
    ('oversampling', None),
    ('model', None)
])
