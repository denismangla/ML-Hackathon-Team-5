from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
OWNERS={'logistic_regression':'DENIS MANGLA | PES2UG24CS151','gaussian_nb':'DENIS MANGLA | PES2UG24CS151','hist_gradient_boosting':'GURUBELLI YEKAMBAR ESHWAR RAO | PES2UG24CS173','random_forest':'DIVYANSHI | PES2UG24CS161','knn':'CH SOWMYA VAISHNAVI | PES2UG24CS126'}
def make_model(name,params=None):
 params=params or {}
 if name=='logistic_regression':
  c=dict(C=1.,max_iter=500,random_state=42);c.update(params);return make_pipeline(StandardScaler(),LogisticRegression(**c))
 if name=='gaussian_nb':return make_pipeline(StandardScaler(),GaussianNB(**params))
 if name=='knn':
  c=dict(n_neighbors=7,weights='distance',algorithm='kd_tree',n_jobs=2);c.update(params);return make_pipeline(StandardScaler(),KNeighborsClassifier(**c))
 if name=='random_forest':
  c=dict(n_estimators=80,max_depth=16,min_samples_leaf=3,n_jobs=2,random_state=42);c.update(params);return RandomForestClassifier(**c)
 if name=='hist_gradient_boosting':
  c=dict(max_iter=100,max_leaf_nodes=15,learning_rate=.1,l2_regularization=1.,random_state=42,early_stopping=False);c.update(params);return HistGradientBoostingClassifier(**c)
 raise ValueError(name)
