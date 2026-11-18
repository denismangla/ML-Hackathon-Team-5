from sklearn.ensemble import RandomForestClassifier

def build(**params):
    settings = dict(n_estimators=80, max_depth=16, min_samples_leaf=3, n_jobs=2, random_state=42)
    settings.update(params)
    return RandomForestClassifier(**settings)
