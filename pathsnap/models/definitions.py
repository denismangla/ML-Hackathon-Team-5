from . import logistic_regression, gaussian_nb, knn, random_forest, hist_gradient_boosting

OWNERS={'logistic_regression':'DENIS MANGLA | PES2UG24CS151','gaussian_nb':'DENIS MANGLA | PES2UG24CS151','hist_gradient_boosting':'GURUBELLI YEKAMBAR ESHWAR RAO | PES2UG24CS173','random_forest':'DIVYANSHI | PES2UG24CS161','knn':'CH SOWMYA VAISHNAVI | PES2UG24CS126'}

BUILDERS = {
    'logistic_regression': logistic_regression.build,
    'gaussian_nb': gaussian_nb.build,
    'knn': knn.build,
    'random_forest': random_forest.build,
    'hist_gradient_boosting': hist_gradient_boosting.build,
}

def make_model(name, params=None):
    if name not in BUILDERS:
        raise ValueError(name)
    return BUILDERS[name](**(params or {}))
