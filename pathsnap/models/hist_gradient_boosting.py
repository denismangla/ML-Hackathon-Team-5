from sklearn.ensemble import HistGradientBoostingClassifier

def build(**params):
    settings = dict(max_iter=100, max_leaf_nodes=15, learning_rate=.1, l2_regularization=1.,
                    random_state=42, early_stopping=False)
    settings.update(params)
    return HistGradientBoostingClassifier(**settings)
