from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

all_models = {
    'logistic_regression':LogisticRegression(n_jobs=1)
}
