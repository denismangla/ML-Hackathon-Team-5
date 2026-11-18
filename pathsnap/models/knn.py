from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def build(**params):
    settings = dict(n_neighbors=7, weights='distance', algorithm='kd_tree', n_jobs=2)
    settings.update(params)
    return make_pipeline(StandardScaler(), KNeighborsClassifier(**settings))
