from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def build(**params):
    settings = dict(C=1., max_iter=500, random_state=42)
    settings.update(params)
    return make_pipeline(StandardScaler(), LogisticRegression(**settings))
