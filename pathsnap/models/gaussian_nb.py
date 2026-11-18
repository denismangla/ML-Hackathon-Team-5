from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def build(**params):
    return make_pipeline(StandardScaler(), GaussianNB(**params))
